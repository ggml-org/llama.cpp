#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-kompute.h"

// These are generated at build time by cmake custom command
#include "shaderop_scale.h"
#include "shaderop_scale_8.h"
#include "shaderop_add.h"
#include "shaderop_addrow.h"
#include "shaderop_mul.h"
#include "shaderop_silu.h"
#include "shaderop_relu.h"
#include "shaderop_gelu.h"
#include "shaderop_softmax.h"
#include "shaderop_norm.h"
#include "shaderop_rmsnorm.h"
#include "shaderop_diagmask.h"
#include "shaderop_mul_mat_f16.h"
#include "shaderop_mul_mat_q8_0.h"
#include "shaderop_mul_mat_q4_0.h"
#include "shaderop_mul_mat_q4_1.h"
#include "shaderop_mul_mat_q6_k.h"
#include "shaderop_mul_mat_mat_f32.h"
#include "shaderop_getrows_f32.h"
#include "shaderop_getrows_f16.h"
#include "shaderop_getrows_q4_0.h"
#include "shaderop_getrows_q4_1.h"
#include "shaderop_getrows_q6_k.h"
#include "shaderop_rope_f16.h"
#include "shaderop_rope_f32.h"
#include "shaderop_cpy_f16_f16.h"
#include "shaderop_cpy_f16_f32.h"
#include "shaderop_cpy_f32_f16.h"
#include "shaderop_cpy_f32_f32.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <kompute/Kompute.hpp>
#include <vulkan/vulkan.hpp>

#ifdef __linux__
#include <cstdlib> // for setenv
#endif

#define QK4_0 32
#define QR4_0 2
#define QK4_1 32
#define QK_NL 16

typedef ggml_fp16_t half;

static std::string ggml_kompute_format_name(int device) {
    return "Kompute" + std::to_string(device);
}

struct ggml_kompute_context {
    int device;
    std::string name;
    std::shared_ptr<vk::DescriptorPool> pool;

    ggml_kompute_context(int device)
        : device(device), name(ggml_kompute_format_name(device)) {}
};

// FIXME: It would be good to consolidate the kompute manager and the kompute context into one object
// and consolidate the init functions and simplify object lifetime management. As it currently stands,
// we *have* to have the kompute manager no matter what for device discovery, but the kompute context
// is only created when a device is set and vulkan is explicitly turned on.
static ggml_kompute_context *s_kompute_context = nullptr;

class kompute_manager {
    kp::Manager *s_mgr = nullptr;

public:
    kp::Manager *operator()() {
        if (s_mgr && !s_mgr->hasInstance()) {
            destroy();
        }
        if (!s_mgr) {
            s_mgr = new kp::Manager;
        }
        return s_mgr;
    }

    void destroy() {
        delete s_mgr;
        s_mgr = nullptr;
    }
};

static kompute_manager komputeManager;

struct ggml_vk_memory {
    void *data = nullptr;
    size_t size = 0;
    vk::DeviceMemory *primaryMemory = nullptr;
    vk::Buffer *primaryBuffer = nullptr;
    vk::DeviceMemory *stagingMemory = nullptr;
    vk::Buffer *stagingBuffer = nullptr;
};

#ifdef __linux__
__attribute__((constructor))
static void enable_sam() {
    setenv("RADV_PERFTEST", "sam", false);
}
#endif

static bool ggml_vk_checkPhysicalDeviceFeatures(vk::PhysicalDevice physical_device) {
    vk::PhysicalDeviceFeatures availableFeatures;
    physical_device.getFeatures(&availableFeatures);

    if (!availableFeatures.shaderInt16)
        return false;

    vk::PhysicalDeviceVulkan11Features availableFeatures11;
    vk::PhysicalDeviceVulkan12Features availableFeatures12;

    availableFeatures11.pNext = &availableFeatures12;
    availableFeatures12.pNext = nullptr;

    vk::PhysicalDeviceFeatures2 features2;
    features2.pNext = &availableFeatures11;

    physical_device.getFeatures2(&features2);

    if (!availableFeatures11.uniformAndStorageBuffer16BitAccess ||
        !availableFeatures11.storageBuffer16BitAccess) {
        return false;
    }

    if (!availableFeatures12.storageBuffer8BitAccess ||
        !availableFeatures12.uniformAndStorageBuffer8BitAccess ||
        !availableFeatures12.shaderFloat16 ||
        !availableFeatures12.shaderInt8) {
        return false;
    }

    return true;
}

static const char * ggml_vk_getVendorName(uint32_t vendorID) {
    switch (vendorID) {
        case 0x10DE:
            return "nvidia";
        case 0x1002:
            return "amd";
        case 0x8086:
            return "intel";
        default:
            return "unknown";
    }
}

static std::vector<ggml_vk_device> ggml_vk_available_devices_internal(size_t memoryRequired) {
    std::vector<ggml_vk_device> results;
    if (!komputeManager()->hasVulkan() || !komputeManager()->hasInstance())
        return results;

    std::vector<vk::PhysicalDevice> physical_devices;
    try {
        physical_devices = komputeManager()->listDevices();
    } catch (vk::SystemError & err) {
        std::cerr << __func__ << ": ignoring Vulkan exception: " << err.what() << "\n";
        return results;
    }

    uint32_t deviceCount = physical_devices.size();
    if (deviceCount == 0)
        return results;

    std::unordered_map<std::string, size_t> count_by_name;

    for (uint32_t i = 0; i < deviceCount; i++) {
        const auto & physical_device = physical_devices[i];

        VkPhysicalDeviceProperties dev_props = physical_device.getProperties();
        VkPhysicalDeviceMemoryProperties memoryProperties = physical_device.getMemoryProperties();
        const uint32_t major = VK_VERSION_MAJOR(dev_props.apiVersion);
        const uint32_t minor = VK_VERSION_MINOR(dev_props.apiVersion);
        if (major < 1 || minor < 2)
            continue;

        if (!ggml_vk_checkPhysicalDeviceFeatures(physical_device))
            continue;

        size_t heapSize = 0;
        for (uint32_t j = 0; j < memoryProperties.memoryHeapCount; ++j) {
            VkMemoryHeap heap = memoryProperties.memoryHeaps[j];
            if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                heapSize = heap.size;
                break;
            }
        }

        if (heapSize < memoryRequired)
            continue;

        auto ext_props = physical_device.enumerateDeviceExtensionProperties();
        bool has_maintenance4 = false;

        // Check if maintenance4 is supported
        for (const auto & properties : ext_props) {
            if (strcmp("VK_KHR_maintenance4", properties.extensionName) == 0) {
                has_maintenance4 = true;
            }
        }

        vk::PhysicalDeviceSubgroupProperties subgroup_props;
        vk::PhysicalDeviceProperties2 dev_props2;
        vk::PhysicalDeviceMaintenance3Properties dev_props3;
        vk::PhysicalDeviceMaintenance4Properties dev_props4;
        dev_props2.pNext = &dev_props3;
        dev_props3.pNext = &subgroup_props;
        if (has_maintenance4) {
            subgroup_props.pNext = &dev_props4;
        }
        physical_device.getProperties2(&dev_props2);

        if (subgroup_props.subgroupSize < 32)
            continue;

        ggml_vk_device d;
        d.index = i;
        d.type = dev_props.deviceType;
        d.heapSize = heapSize;
        d.vendor = strdup(ggml_vk_getVendorName(dev_props.vendorID));
        d.subgroupSize = subgroup_props.subgroupSize;
        d.bufferAlignment = dev_props.limits.minStorageBufferOffsetAlignment;

        if (has_maintenance4) {
            d.maxAlloc = std::min(dev_props3.maxMemoryAllocationSize, dev_props4.maxBufferSize);
        } else {
            d.maxAlloc = dev_props3.maxMemoryAllocationSize;
        }

        std::string name(dev_props.deviceName);
        size_t n_idx = ++count_by_name[name];
        if (n_idx > 1) {
            name += " (" + std::to_string(n_idx) + ")";
        }
        d.name = strdup(name.c_str());

        results.push_back(d);
    }

    std::stable_sort(results.begin(), results.end(),
        [](const ggml_vk_device& lhs, const ggml_vk_device& rhs) -> bool {
            if (lhs.type != rhs.type) {
                if (lhs.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return true;
                if (rhs.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return false;

                if (lhs.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return true;
                if (rhs.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return false;
            }
            return lhs.heapSize < rhs.heapSize;
        }
    );

    return results;
}

// public API returns a C-style array
ggml_vk_device * ggml_vk_available_devices(size_t memoryRequired, size_t * count) {
    auto devices = ggml_vk_available_devices_internal(memoryRequired);
    *count = devices.size();
    if (devices.empty()) {
        return nullptr;
    }

    size_t nbytes = sizeof (ggml_vk_device) * (devices.size());
    auto * arr = static_cast<ggml_vk_device *>(malloc(nbytes));
    memcpy(arr, devices.data(), nbytes);
    return arr;
}

static void ggml_vk_filterByVendor(std::vector<ggml_vk_device>& devices, const std::string& targetVendor) {
    devices.erase(
        std::remove_if(devices.begin(), devices.end(),
            [&targetVendor](const ggml_vk_device& device) {
                return device.vendor != targetVendor;
            }),
        devices.end()
    );
}

static void ggml_vk_filterByName(std::vector<ggml_vk_device>& devices, const std::string& targetName) {
    devices.erase(
        std::remove_if(devices.begin(), devices.end(),
            [&targetName](const ggml_vk_device& device) {
                return device.name != targetName;
            }),
        devices.end()
    );
}

static bool ggml_vk_get_device(ggml_vk_device * device, size_t memoryRequired, const std::string & name) {
    if (name.empty())
        return false;

    auto devices = ggml_vk_available_devices_internal(memoryRequired);
    if (name == "amd" || name == "nvidia" || name == "intel") {
        ggml_vk_filterByVendor(devices, name);
    } else if (name != "gpu") {
        ggml_vk_filterByName(devices, name);
    }

    if (devices.empty())
        return false;

    *device = devices.front();
    return true;
}

bool ggml_vk_get_device(ggml_vk_device * device, size_t memoryRequired, const char * name) {
    return ggml_vk_get_device(device, memoryRequired, std::string(name));
}

bool ggml_vk_has_vulkan() {
    return komputeManager()->hasVulkan();
}

bool ggml_vk_has_device() {
    return komputeManager()->hasDevice();
}

ggml_vk_device ggml_vk_current_device() {
    if (!komputeManager()->hasDevice())
        return ggml_vk_device();

    auto devices = ggml_vk_available_devices_internal(0);
    ggml_vk_filterByName(devices, komputeManager()->physicalDevice()->getProperties().deviceName.data());
    GGML_ASSERT(!devices.empty());
    return devices.front();
}

static
void ggml_vk_allocate_descriptor_pool(struct ggml_kompute_context * ctx, size_t size) {
    std::vector<vk::DescriptorPoolSize> descriptorPoolSizes = {
        vk::DescriptorPoolSize(
          vk::DescriptorType::eStorageBuffer,
          3 * size // Descriptor count is number of possible tensors to pass into an algorithm
          )
    };

    vk::DescriptorPoolCreateInfo descriptorPoolInfo(
      vk::DescriptorPoolCreateFlags(),
      size, // Max sets
      static_cast<uint32_t>(descriptorPoolSizes.size()),
      descriptorPoolSizes.data());

    ctx->pool = std::make_shared<vk::DescriptorPool>();
    vk::Result r = komputeManager()->device()->createDescriptorPool(
      &descriptorPoolInfo, nullptr, ctx->pool.get());
    if (r != vk::Result::eSuccess)
        std::cerr << "Error allocating descriptor pool" << vk::to_string(r);
}

static
void ggml_vk_free_descriptor_pool(struct ggml_kompute_context * ctx) {
    if (ctx->pool) {
        komputeManager()->device()->destroy(
          *ctx->pool,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
        ctx->pool = nullptr;
    }
}

static
vk::Buffer *ggml_vk_allocate_buffer(size_t size) {
    vk::BufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferSrc |
                             vk::BufferUsageFlagBits::eTransferDst;
    bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::Buffer *vkBuffer = new vk::Buffer;
    vk::Result r = komputeManager()->device()->createBuffer(&bufferCreateInfo, nullptr, vkBuffer);
    if (r != vk::Result::eSuccess)
        std::cerr << "Error allocating buffer " << vk::to_string(r) << std::endl;
    return vkBuffer;
}

static
vk::DeviceMemory *ggml_vk_allocate(size_t size, vk::MemoryPropertyFlags flags, vk::MemoryRequirements requirements, bool *isHostVisible) {

    uint32_t memoryTypeIndex = -1;
    bool memoryTypeIndexFound = false;
    vk::PhysicalDeviceMemoryProperties memoryProperties = komputeManager()->physicalDevice()->getMemoryProperties();
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        const vk::MemoryType &memoryType = memoryProperties.memoryTypes[i];
        const vk::MemoryHeap &memoryHeap = memoryProperties.memoryHeaps[memoryType.heapIndex];
        if (memoryHeap.size < size) {
            continue;
        }

        if (requirements.memoryTypeBits & (1 << i)) {
            if (((memoryProperties.memoryTypes[i]).propertyFlags &
                 flags) == flags) {
                memoryTypeIndex = i;
                memoryTypeIndexFound = true;
                if (isHostVisible && (memoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                    *isHostVisible = true;
                }
                break;
            }
        }
    }
    if (!memoryTypeIndexFound) {
        throw std::runtime_error(
          "Memory type index for buffer creation not found");
    }

    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    vk::DeviceMemory *vkDeviceMemory =  new vk::DeviceMemory;
    vk::Result r = komputeManager()->device()->allocateMemory(&allocInfo, nullptr, vkDeviceMemory);
    if (r != vk::Result::eSuccess) {
        std::cerr << "Error allocating memory " << vk::to_string(r) << std::endl;
        throw std::runtime_error("Error allocating vulkan memory.");
    }
    return vkDeviceMemory;
}

static size_t ggml_vk_aligned_offset(ggml_backend_buffer_t buffer, size_t offset) {
    size_t minStorageBufferOffsetAlignment = ggml_backend_buffer_get_alignment(buffer);

    // If offset is already aligned, return it directly
    if (offset % minStorageBufferOffsetAlignment == 0) {
        return offset;
    }

    // Otherwise, return the largest multiple of minStorageBufferOffsetAlignment less than offset
    return (offset / minStorageBufferOffsetAlignment) * minStorageBufferOffsetAlignment;
}

static ggml_vk_memory ggml_vk_allocate(size_t size) {
    ggml_vk_memory memory;
    bool isHostVisible = false;
    {
        memory.primaryBuffer = ggml_vk_allocate_buffer(size);
        vk::MemoryRequirements memoryRequirements = komputeManager()->device()->getBufferMemoryRequirements(*memory.primaryBuffer);
        vk::MemoryPropertyFlags memoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        memory.primaryMemory = ggml_vk_allocate(size, memoryPropertyFlags, memoryRequirements, &isHostVisible);
        komputeManager()->device()->bindBufferMemory(*memory.primaryBuffer, *memory.primaryMemory, 0);
        if (isHostVisible) {
            vk::Result r = komputeManager()->device()->mapMemory(*memory.primaryMemory, 0, size, vk::MemoryMapFlags(), &memory.data);
            if (r != vk::Result::eSuccess)
                std::cerr << "Error mapping memory" << vk::to_string(r);
        }
    }

    if (!isHostVisible) {
        memory.stagingBuffer = ggml_vk_allocate_buffer(size);
        vk::MemoryRequirements memoryRequirements = komputeManager()->device()->getBufferMemoryRequirements(*memory.stagingBuffer);
        vk::MemoryPropertyFlags memoryPropertyFlags = vk::MemoryPropertyFlagBits::eHostVisible |
                                                      vk::MemoryPropertyFlagBits::eHostCoherent |
                                                      vk::MemoryPropertyFlagBits::eHostCached;
        memory.stagingMemory = ggml_vk_allocate(size, memoryPropertyFlags, memoryRequirements, &isHostVisible);
        komputeManager()->device()->bindBufferMemory(*memory.stagingBuffer, *memory.stagingMemory, 0);
        vk::Result r = komputeManager()->device()->mapMemory(*memory.stagingMemory, 0, size, vk::MemoryMapFlags(), &memory.data);
        if (r != vk::Result::eSuccess)
            std::cerr << "Error mapping memory" << vk::to_string(r);
    }

    memory.size = size;
    return memory;
}

static void ggml_vk_free_memory(ggml_vk_memory &memory)
{
    komputeManager()->device()->destroy(
      *memory.primaryBuffer,
      (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    if (memory.stagingBuffer) {
        komputeManager()->device()->destroy(
          *memory.stagingBuffer,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    }
    komputeManager()->device()->freeMemory(
      *memory.primaryMemory,
      (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    if (memory.stagingMemory) {
        komputeManager()->device()->freeMemory(
          *memory.stagingMemory,
          (vk::Optional<const vk::AllocationCallbacks>)nullptr);
    }
}

static const char * ggml_backend_kompute_buffer_type_get_name(ggml_backend_buffer_type_t buft);

static
ggml_vk_memory * ggml_vk_find_tensor(const struct ggml_tensor * t, uint64_t & offset) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    // compatibility with ggml-backend
    GGML_ASSERT(buffer && buffer->buft->iface.get_name == ggml_backend_kompute_buffer_type_get_name);

    ggml_vk_memory * buf_ctx = static_cast<ggml_vk_memory *>(buffer->context);

    const intptr_t ioffs = intptr_t(t->data) - intptr_t(buf_ctx->data);

    GGML_ASSERT(ioffs >= 0 && ioffs + int64_t(ggml_nbytes(t)) <= int64_t(buffer->size));

    offset = uint64_t(ioffs);
    return buf_ctx;
}

static
const std::shared_ptr<kp::Tensor> ggml_vk_get_tensor(const struct ggml_tensor * t, uint32_t * alignedOffset = nullptr) {
    uint64_t originalOffset = 0;
    auto * res = ggml_vk_find_tensor(t, originalOffset);
    if (!res) {
        static std::shared_ptr<kp::Tensor> nullTensor = nullptr;
        return nullTensor;
    }

    // Create a tensor whose memory will be composed of our buffers at the correct offset
    const size_t nelements = ggml_nelements(t);
    size_t nbytes = ggml_nbytes(t);

    size_t vulkanOffset = ggml_vk_aligned_offset(t->buffer, originalOffset);
    if (alignedOffset) {
        *alignedOffset = originalOffset - vulkanOffset;
        nbytes += *alignedOffset;
    }

    return komputeManager()->tensor(
        t->data,
        nelements,
        nbytes, kp::Tensor::TensorDataTypes::eFloat,
        res->primaryMemory, res->primaryBuffer,
        res->stagingMemory, res->stagingBuffer,
        vulkanOffset);
}

static std::vector<uint32_t> getSpirvShader(const unsigned char* rawData, size_t size) {
    if (size % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Invalid size: must be divisible by sizeof(uint32_t)");
    }

    const uint32_t* data_ptr = reinterpret_cast<const uint32_t*>(rawData);
    size_t count = size / sizeof(uint32_t);
    return std::vector<uint32_t>(data_ptr, data_ptr + count);
}

inline static
uint32_t safe_divide(uint32_t a, uint32_t b) {
    if (b <= 1) {
        return a;
    }
    if ((a % b) != 0) {
        fprintf(stderr, "((%u %% %u) == %u) != 0\n", a, b, a % b);
        GGML_ABORT("safe_divide result would've had remainder");
    }
    return a / b;
}

static void ggml_vk_add(
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
    int32_t nb00, int32_t nb01, int32_t nb02, int32_t nb03,
    int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,
    int32_t nb10, int32_t nb11, int32_t nb12, int32_t nb13,
    int32_t ne0,
    int32_t nb0,  int32_t nb1,  int32_t nb2,  int32_t nb3
) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_add_comp_spv,
        kp::shader_data::op_add_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00;
        int32_t nb00, nb01, nb02, nb03;
        int32_t ne10, ne11, ne12, ne13;
        int32_t nb10, nb11, nb12, nb13;
        int32_t ne0;
        int32_t nb0, nb1, nb2, nb3;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00,
        nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
        nb10, nb11, nb12, nb13,
        ne0,
        nb0, nb1, nb2, nb3
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_addrow(kp::Sequence& seq,
                 const std::shared_ptr<kp::Tensor>& inA,
                 const std::shared_ptr<kp::Tensor>& inB,
                 const std::shared_ptr<kp::Tensor>& out,
                 uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                 uint32_t size, uint32_t row = 0) {

    const static auto spirv = getSpirvShader(kp::shader_data::op_addrow_comp_spv,
        kp::shader_data::op_addrow_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        uint32_t row;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        row
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_mul(
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
    int32_t nb00, int32_t nb01, int32_t nb02, int32_t nb03,
    int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,
    int32_t nb10, int32_t nb11, int32_t nb12, int32_t nb13,
    int32_t ne0,
    int32_t nb0,  int32_t nb1,  int32_t nb2,  int32_t nb3
) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_comp_spv,
        kp::shader_data::op_mul_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00;
        int32_t nb00, nb01, nb02, nb03;
        int32_t ne10, ne11, ne12, ne13;
        int32_t nb10, nb11, nb12, nb13;
        int32_t ne0;
        int32_t nb0, nb1, nb2, nb3;
    } const pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00,
        nb00, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
        nb10, nb11, nb12, nb13,
        ne0,
        nb0, nb1, nb2, nb3
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_scale(kp::Sequence& seq,
                   const std::shared_ptr<kp::Tensor>& in,
                   const std::shared_ptr<kp::Tensor>& out,
                   uint32_t inOff, uint32_t outOff,
                   uint32_t size, float scale) {
    const static auto spirv_1 = getSpirvShader(
        kp::shader_data::op_scale_comp_spv, kp::shader_data::op_scale_comp_spv_len
    );
    const static auto spirv_8 = getSpirvShader(
        kp::shader_data::op_scale_8_comp_spv, kp::shader_data::op_scale_8_comp_spv_len
    );

    struct PushConstants {
        uint32_t inOff, outOff;
        float scale;
    } const pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        scale
    };

    const auto * spirv = &spirv_1;
    std::string name(__func__);
    if (size % 8 == 0) {
        size /= 8;
        name += "_8";
        spirv = &spirv_8;
    }

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(name, s_kompute_context->pool.get(), {in, out}, *spirv, {size}, {}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_xxlu(
    const std::vector<uint32_t>& spirv, const char * suffix, kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& in,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inOff, uint32_t outOff,
    uint32_t size
) {
    struct PushConstants {
        uint32_t inOff, outOff;
    } const pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
    };

    auto name = std::string(__func__) + "_" + suffix;
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(name, s_kompute_context->pool.get(), {in, out}, spirv, {size}, {}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
static void ggml_vk_silu(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_silu_comp_spv,
        kp::shader_data::op_silu_comp_spv_len);

    ggml_vk_xxlu(spirv, "silu", std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_relu(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_relu_comp_spv,
        kp::shader_data::op_relu_comp_spv_len);

    ggml_vk_xxlu(spirv, "relu", std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_gelu(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_gelu_comp_spv,
        kp::shader_data::op_gelu_comp_spv_len);

    ggml_vk_xxlu(spirv, "gelu", std::forward<Args>(args)...);
}

static void ggml_vk_soft_max(
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t ne01, int32_t ne02, uint32_t ne03,
    float scale
) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_softmax_comp_spv,
        kp::shader_data::op_softmax_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne01, ne02;
        float scale;
        int32_t mask;
    } pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne01, ne02,
        scale,
        bool(inB)
    };

    auto & inB_ = inB ? inB : inA;

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        // FIXME: The softmax kernel needs to be fixed to use the subgroupsize which can vary by device
        const uint32_t local_x = 32;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB_, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB_, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_norm_(
    const std::vector<uint32_t>& spirv, const char * suffix, kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& in,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inOff, uint32_t outOff,
    int32_t ne00, int32_t nb01,
    int32_t nrows, float epsilon
) {
    GGML_ASSERT(nb01%sizeof(float) == 0);
    GGML_ASSERT(ne00%sizeof(float) == 0);

    struct PushConstants {
        uint32_t inOff, outOff;
        uint32_t ne00, nb01;
        float eps;
    } pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        (uint32_t)ne00, (uint32_t)nb01, epsilon
    };

    auto name = std::string(__func__) + "_" + suffix;
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(name, s_kompute_context->pool.get(), {in, out}, spirv, {(uint32_t)nrows}, {}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({(uint32_t)nrows});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
static void ggml_vk_norm(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_norm_comp_spv,
        kp::shader_data::op_norm_comp_spv_len);

    ggml_vk_norm_(spirv, "norm", std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_rms_norm(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_rmsnorm_comp_spv,
        kp::shader_data::op_rmsnorm_comp_spv_len);

    ggml_vk_norm_(spirv, "rms", std::forward<Args>(args)...);
}

static void ggml_vk_diag_mask_inf(kp::Sequence& seq,
                           const std::shared_ptr<kp::Tensor>& in,
                           const std::shared_ptr<kp::Tensor>& out,
                           uint32_t inOff, uint32_t outOff,
                           uint32_t n_past,
                           int32_t ne00, int32_t ne01, int32_t ne02) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_diagmask_comp_spv,
        kp::shader_data::op_diagmask_comp_spv_len);

    struct PushConstants {
        uint32_t inOff, outOff;
        uint32_t n_past;
        int32_t ne00, ne01;
    } pushConsts {
        safe_divide(inOff, 4), safe_divide(outOff, 4),
        n_past,
        ne00, ne01
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__))
        s_algo = komputeManager()->algorithm<float, PushConstants>(__func__, s_kompute_context->pool.get(), {in, out}, spirv, {unsigned(ne00), unsigned(ne01), unsigned(ne02)}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({unsigned(ne00), unsigned(ne01), unsigned(ne02)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_mul_mat_f16(
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t ne01, int32_t ne02,
    uint32_t nb00, uint32_t nb01, uint32_t nb02,
    int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,
    uint32_t nb10, uint32_t nb11, uint32_t nb12,
    int32_t ne0, int32_t ne1,
    uint32_t r2, uint32_t r3
) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_f16_comp_spv,
        kp::shader_data::op_mul_mat_f16_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne01, ne02;
        uint32_t nb00, nb01, nb02;
        int32_t ne10, ne11, ne12;
        uint32_t nb10, nb11, nb12;
        int32_t ne0, ne1;
        uint32_t r2, r3;
    } pushConsts {
        safe_divide(inAOff, 2), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne01, ne02,
        nb00, nb01, nb02,
        ne10, ne11, ne12,
        nb10, nb11, nb12,
        ne0, ne1,
        r2, r3
    };

    const unsigned ny = unsigned((ne11 + 4 - 1)/4);

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize * 2;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned(ne01), ny, unsigned(ne12*ne13)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01), ny, unsigned(ne12*ne13)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_mul_mat_mat_f32(kp::Sequence& seq,
                         const std::shared_ptr<kp::Tensor>& inA,
                         const std::shared_ptr<kp::Tensor>& inB,
                         const std::shared_ptr<kp::Tensor>& out,
                         uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
                         int32_t ne00, int32_t ne01, int32_t ne02,
                         uint32_t nb01, uint32_t nb02,
                         int32_t ne11, int32_t ne12,
                         uint32_t nb11, uint32_t nb12,
                         uint32_t nb1, uint32_t nb2) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_mat_f32_comp_spv,
        kp::shader_data::op_mul_mat_mat_f32_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne01, ne02, ne11, ne12;
        uint32_t nb01, nb02;
        uint32_t nb11, nb12;
        uint32_t nb1, nb2;
    } pushConsts {
        safe_divide(inAOff, 4), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne01, ne02, ne11, ne12,
        nb01, nb02, nb11, nb12,
        nb1, nb2
    };

    const uint32_t local_x = ggml_vk_current_device().subgroupSize;
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(),
        {inA, inB, out}, spirv,
        {unsigned(ne01),
         unsigned(ne11),
         unsigned(std::max(ne12, ne02))
         },
        {local_x},
        {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01),
                              unsigned(ne11),
                              unsigned(std::max(ne12, ne02)),
                              });
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_mul_mat_impl(
    const std::vector<uint32_t>& spirv, const char * suffix, uint32_t block_size, kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t ne01, int32_t ne02,
    int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,
    int32_t ne0, int32_t ne1,
    uint32_t r2, uint32_t r3
) {
    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne01, ne02;
        int32_t ne10, ne12;
        int32_t ne0, ne1;
        uint32_t r2, r3;
    } pushConsts {
        safe_divide(inAOff, block_size), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne01, ne02,
        ne10, ne12,
        ne0, ne1,
        r2, r3
    };

    auto name = std::string(__func__) + "_" + suffix;
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize * 2;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(name, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned((ne01 + 7)/8), unsigned(ne11), unsigned(ne12*ne13)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned((ne01 + 7)/8), unsigned(ne11), unsigned(ne12*ne13)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
static void ggml_vk_mul_mat_q4_0(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q4_0_comp_spv,
        kp::shader_data::op_mul_mat_q4_0_comp_spv_len);

    ggml_vk_mul_mat_impl(spirv, "q4_0", 1/*We access blocks unaligned*/, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_mul_mat_q4_1(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q4_1_comp_spv,
        kp::shader_data::op_mul_mat_q4_1_comp_spv_len);

    ggml_vk_mul_mat_impl(spirv, "q4_1", 1/*We access blocks unaligned*/, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_mul_mat_q8_0(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q8_0_comp_spv,
        kp::shader_data::op_mul_mat_q8_0_comp_spv_len);

    ggml_vk_mul_mat_impl(spirv, "q8_0", 1/*We access blocks unaligned*/, std::forward<Args>(args)...);
}

static void ggml_vk_mul_mat_q6_k(
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t ne10, int32_t ne0, int32_t ne1,
    int32_t ne01, int32_t ne11, int32_t ne12, int32_t ne02
) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_mul_mat_q6_k_comp_spv,
        kp::shader_data::op_mul_mat_q6_k_comp_spv_len);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, ne10, ne0, ne1, ne01, gqa;
    } pushConsts {
        inAOff, safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, ne10, ne0, ne1, ne01, ne12/ne02
    };

    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(__func__)) {
        const uint32_t local_x = ggml_vk_current_device().subgroupSize * 2;
        s_algo = komputeManager()->algorithm<uint32_t, PushConstants>(__func__, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {unsigned((ne01 + 1)/2), unsigned(ne11), unsigned(ne12)}, {local_x}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(__func__);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned((ne01 + 1)/2), unsigned(ne11), unsigned(ne12)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_get_rows(
    const std::vector<uint32_t>& spirv,
    const char * suffix,
    unsigned element_size, unsigned qk,
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    int32_t ne00, int32_t nb01, int32_t nb1,
    uint32_t size
) {
    GGML_ASSERT(nb01%element_size == 0);
    GGML_ASSERT(nb1%sizeof(float) == 0);
    if (qk) GGML_ASSERT(ne00%qk == 0);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t ne00, nb01, nb1;
    } pushConsts {
        safe_divide(inAOff, element_size), safe_divide(inBOff, 4), safe_divide(outOff, 4),
        ne00, nb01, nb1
    };

    auto name = std::string(__func__) + "_" + suffix;
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(name, s_kompute_context->pool.get(), {inA, inB, out}, spirv, {size}, {}, {pushConsts});
    } else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({size});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
static void ggml_vk_get_rows_f32(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_f32_comp_spv,
        kp::shader_data::op_getrows_f32_comp_spv_len);

    ggml_vk_get_rows(spirv, "f32", sizeof(float), 0, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_get_rows_f16(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_f16_comp_spv,
        kp::shader_data::op_getrows_f16_comp_spv_len);

    ggml_vk_get_rows(spirv, "f16", sizeof(half), 0, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_get_rows_q4_0(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_q4_0_comp_spv,
        kp::shader_data::op_getrows_q4_0_comp_spv_len);

    ggml_vk_get_rows(spirv, "q4_0", 1/*We access blocks unaligned*/, QK4_0, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_get_rows_q4_1(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_q4_1_comp_spv,
        kp::shader_data::op_getrows_q4_1_comp_spv_len);

    ggml_vk_get_rows(spirv, "q4_1", 1/*We access blocks unaligned*/, QK4_1, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_get_rows_q6_k(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_getrows_q6_k_comp_spv,
        kp::shader_data::op_getrows_q6_k_comp_spv_len);
    ggml_vk_get_rows(spirv, "q6_k", 1/*We access blocks unaligned*/, QK_NL, std::forward<Args>(args)...);
}

static void ggml_vk_rope(
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& inA,
    const std::shared_ptr<kp::Tensor>& inB,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inAOff, uint32_t inBOff, uint32_t outOff,
    ggml_type src0t, int32_t n_dims, int32_t mode, int32_t n_ctx_orig,
    float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow,
    int32_t ne01, int32_t ne02, int32_t ne03,
    uint32_t nb00, uint32_t nb01, uint32_t nb02, uint32_t nb03,
    int32_t ne0,
    uint32_t nb0, uint32_t nb1, uint32_t nb2, uint32_t nb3
) {
    GGML_ASSERT(src0t == GGML_TYPE_F16 || src0t == GGML_TYPE_F32);

    static const auto spirv_f16 = getSpirvShader(
        kp::shader_data::op_rope_f16_comp_spv, kp::shader_data::op_rope_f16_comp_spv_len
    );
    static const auto spirv_f32 = getSpirvShader(
        kp::shader_data::op_rope_f32_comp_spv, kp::shader_data::op_rope_f32_comp_spv_len
    );

    int type_size = src0t == GGML_TYPE_F16 ? 2 : 4;

    GGML_ASSERT(nb03 % type_size == 0);
    GGML_ASSERT(nb02 % type_size == 0);
    GGML_ASSERT(nb01 % type_size == 0);
    GGML_ASSERT(nb00 % type_size == 0);
    GGML_ASSERT(nb3  % type_size == 0);
    GGML_ASSERT(nb2  % type_size == 0);
    GGML_ASSERT(nb1  % type_size == 0);
    GGML_ASSERT(nb0  % type_size == 0);

    struct PushConstants {
        uint32_t inAOff, inBOff, outOff;
        int32_t n_dims, mode, n_ctx_orig;
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        uint32_t nb00, nb01, nb02, nb03;
        int32_t ne0;
        uint32_t nb0, nb1, nb2, nb3;
    } pushConsts {
        safe_divide(inAOff, type_size), safe_divide(inBOff, 4), safe_divide(outOff, type_size),
        n_dims, mode, n_ctx_orig,
        freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
        nb00, nb01, nb02, nb03,
        ne0,
        nb0, nb1, nb2, nb3
    };

    auto name = std::string(__func__) + (src0t == GGML_TYPE_F16 ? "_f16" : "_f32");
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name)) {
        s_algo = komputeManager()->algorithm<float, PushConstants>(
            name, s_kompute_context->pool.get(), {inA, inB, out},
            src0t == GGML_TYPE_F16 ? spirv_f16 : spirv_f32,
            {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts}
        );
    } else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({inA, inB, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

static void ggml_vk_cpy(
    const std::vector<uint32_t>& spirv,
    uint32_t in_element_size, uint32_t out_element_size,
    kp::Sequence& seq,
    const std::shared_ptr<kp::Tensor>& in,
    const std::shared_ptr<kp::Tensor>& out,
    uint32_t inOff, uint32_t outOff,
    int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
    uint32_t nb00, uint32_t nb01, uint32_t nb02, uint32_t nb03,
    int32_t ne0, int32_t ne1, int32_t ne2,
    uint32_t nb0, uint32_t nb1, uint32_t nb2, uint32_t nb3
) {
    struct PushConstants {
        uint32_t inOff, outOff;
        int32_t ne00, ne01, ne02;
        uint32_t nb00, nb01, nb02, nb03;
        int32_t ne0, ne1, ne2;
        uint32_t nb0, nb1, nb2, nb3;
    } pushConsts {
        safe_divide(inOff, in_element_size), safe_divide(outOff, out_element_size),
        ne00, ne01, ne02,
        nb00, nb01, nb02, nb03,
        ne0, ne1, ne2,
        nb0, nb1, nb2, nb3
    };

    std::string name = std::string(__func__)
                       + "_i_" + std::to_string(in_element_size)
                       + "_o_" + std::to_string(out_element_size);
    std::shared_ptr<kp::Algorithm> s_algo = nullptr;
    if (!komputeManager()->hasAlgorithm(name))
        s_algo = komputeManager()->algorithm<float, PushConstants>(name, s_kompute_context->pool.get(), {in, out}, spirv, {unsigned(ne01), unsigned(ne02), unsigned(ne03)}, {}, {pushConsts});
    else {
        s_algo = komputeManager()->getAlgorithm(name);
        s_algo->setTensors({in, out});
        s_algo->setWorkgroup({unsigned(ne01), unsigned(ne02), unsigned(ne03)});
        s_algo->setPushConstants<PushConstants>({pushConsts});
        s_algo->updateDescriptors(s_kompute_context->pool.get());
    }
    seq.record<kp::OpAlgoDispatch>(s_algo);
}

template <typename... Args>
static void ggml_vk_cpy_f32_f16(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f32_f16_comp_spv,
        kp::shader_data::op_cpy_f32_f16_comp_spv_len);
    ggml_vk_cpy(spirv, 4, 2, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_cpy_f32_f32(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f32_f32_comp_spv,
        kp::shader_data::op_cpy_f32_f32_comp_spv_len);
    ggml_vk_cpy(spirv, 4, 4, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_cpy_f16_f16(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f16_f16_comp_spv,
        kp::shader_data::op_cpy_f16_f16_comp_spv_len);
    ggml_vk_cpy(spirv, 2, 2, std::forward<Args>(args)...);
}

template <typename... Args>
static void ggml_vk_cpy_f16_f32(Args&&... args) {
    const static auto spirv = getSpirvShader(kp::shader_data::op_cpy_f16_f32_comp_spv,
        kp::shader_data::op_cpy_f16_f32_comp_spv_len);
    ggml_vk_cpy(spirv, 2, 4, std::forward<Args>(args)...);
}

static bool ggml_vk_supports_op(const struct ggml_tensor * op) {
    switch (op->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
            break;
        default:
            return false;
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    ;
            }
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_ROPE:
            return true;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    break;
                default:
                    return false;
            }
            switch (op->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    break;
                default:
                    return false;
            }
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return op->ne[3] == 1;
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q6_K:
                    return op->ne[2] == 1 && op->ne[3] == 1;
                default:
                    ;
            }
            return false;
        case GGML_OP_MUL_MAT:
            if (op->src[1]->type != GGML_TYPE_F32 || ggml_is_transposed(op->src[0]) || ggml_is_transposed(op->src[1]))
                return false;

            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_Q6_K:
                    return op->ne[3] == 1;
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return true;
                default:
                    ;
            }
        default:
            ;
    }
    return false;
}

static void ggml_vk_graph_compute(struct ggml_kompute_context * ctx, struct ggml_cgraph * gf) {
    const int n_seq = 8;

    // FIXME: Figure out if we can somehow optimize the size of the pool... right now we're setting
    // it to the size of the graph, but I think it can be made smaller?
    ggml_vk_allocate_descriptor_pool(ctx, gf->n_nodes);

    std::vector<std::shared_ptr<kp::Sequence>> sequences(n_seq);

    for (auto& sequence : sequences) {
        sequence = komputeManager()->sequence();
    }
    for (int seq_idx = 0; seq_idx < n_seq; ++seq_idx) {
        const int n_nodes_per_seq = (gf->n_nodes + n_seq - 1) / n_seq;

        auto& seq = *sequences[seq_idx];

        const int node_start = (seq_idx + 0) * n_nodes_per_seq;
        const int node_end   = std::min((seq_idx == n_seq - 1) ? gf->n_nodes : (seq_idx + 1) * n_nodes_per_seq, gf->n_nodes);

        bool any_commands_recorded = false;

        for (int i = node_start; i < node_end; ++i) {
            struct ggml_tensor * src0 = gf->nodes[i]->src[0];
            struct ggml_tensor * src1 = gf->nodes[i]->src[1];
            struct ggml_tensor * src2 = gf->nodes[i]->src[2]; GGML_UNUSED(src2);
            struct ggml_tensor * dst = gf->nodes[i];
            GGML_ASSERT(dst->data != nullptr);

            if (ggml_is_empty(dst)) {
                continue;
            }

            switch (dst->op) {
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_PERMUTE:
                    continue; // noop -> next node
                default:
                    break;
            }

            any_commands_recorded = true;

            if (!ggml_vk_supports_op(dst)) {
                 fprintf(stderr, "%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(dst));
                 GGML_ABORT("unsupported op");
            }

            const int32_t ne00 = src0 ? src0->ne[0] : 0;
            const int32_t ne01 = src0 ? src0->ne[1] : 0;
            const int32_t ne02 = src0 ? src0->ne[2] : 0;
            const int32_t ne03 = src0 ? src0->ne[3] : 0;

            const uint32_t nb00 = src0 ? src0->nb[0] : 0;
            const uint32_t nb01 = src0 ? src0->nb[1] : 0;
            const uint32_t nb02 = src0 ? src0->nb[2] : 0;
            const uint32_t nb03 = src0 ? src0->nb[3] : 0;

            const int32_t ne10 = src1 ? src1->ne[0] : 0;
            const int32_t ne11 = src1 ? src1->ne[1] : 0;
            const int32_t ne12 = src1 ? src1->ne[2] : 0;
            const int32_t ne13 = src1 ? src1->ne[3] : 0;

            const uint32_t nb10 = src1 ? src1->nb[0] : 0;
            const uint32_t nb11 = src1 ? src1->nb[1] : 0;
            const uint32_t nb12 = src1 ? src1->nb[2] : 0;
            const uint32_t nb13 = src1 ? src1->nb[3] : 0;

            const int32_t ne0 = dst ? dst->ne[0] : 0;
            const int32_t ne1 = dst ? dst->ne[1] : 0;
            const int32_t ne2 = dst ? dst->ne[2] : 0;
//            const int32_t ne3 = dst ? dst->ne[3] : 0;

            const uint32_t nb0 = dst ? dst->nb[0] : 0;
            const uint32_t nb1 = dst ? dst->nb[1] : 0;
            const uint32_t nb2 = dst ? dst->nb[2] : 0;
            const uint32_t nb3 = dst ? dst->nb[3] : 0;

            const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
            const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
            const enum ggml_type dstt = dst ? dst->type : GGML_TYPE_COUNT;

            const static std::shared_ptr<kp::Tensor> nullTensor = nullptr;
            uint32_t off_src0 = 0;
            uint32_t off_src1 = 0;
            uint32_t off_dst  = 0;
            const std::shared_ptr<kp::Tensor>& id_src0 = src0 ? ggml_vk_get_tensor(src0, &off_src0) : nullTensor;
            const std::shared_ptr<kp::Tensor>& id_src1 = src1 ? ggml_vk_get_tensor(src1, &off_src1) : nullTensor;
            const std::shared_ptr<kp::Tensor>& id_dst  = dst  ? ggml_vk_get_tensor(dst,  &off_dst)  : nullTensor;

            switch (dst->op) {
                case GGML_OP_ADD:
                    {
                        if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
                            // src1 is a row
                            ggml_vk_addrow(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ggml_nelements(dst)/4, ne00);
                        } else {
                            ggml_vk_add(
                                seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                ne00, ne01, ne02, ne03,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                ne0,
                                nb0, nb1, nb2, nb3
                            );
                        }
                    } break;
                case GGML_OP_MUL:
                    {
                        ggml_vk_mul(
                            seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                            ne00, ne01, ne02, ne03,
                            nb00, nb01, nb02, nb03,
                            ne10, ne11, ne12, ne13,
                            nb10, nb11, nb12, nb13,
                            ne0,
                            nb0, nb1, nb2, nb3
                        );
                    } break;
                case GGML_OP_SCALE:
                    {
                        float scale; memcpy(&scale, dst->op_params, sizeof(float));

                        ggml_vk_scale(seq, id_src0, id_dst, off_src0, off_dst, ggml_nelements(dst), scale);
                    } break;
                case GGML_OP_UNARY:
                    {
                        int64_t n = ggml_nelements(dst);
                        GGML_ASSERT(n % 4 == 0);
                        switch (ggml_get_unary_op(gf->nodes[i])) {
                            case GGML_UNARY_OP_SILU:
                                {
                                    ggml_vk_silu(seq, id_src0, id_dst, off_src0, off_dst, n/4);
                                } break;
                            case GGML_UNARY_OP_RELU:
                                {
                                    ggml_vk_relu(seq, id_src0, id_dst, off_src0, off_dst, n/4);
                                } break;
                            case GGML_UNARY_OP_GELU:
                                {
                                    GGML_ASSERT(n % 8 == 0);
                                    ggml_vk_gelu(seq, id_src0, id_dst, off_src0, off_dst, n/8);
                                } break;
                            default:
                                {
                                    fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                    GGML_ABORT("fatal error");
                                }
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                    {
                        float scale;
                        float max_bias;

                        memcpy(&scale,    (float *)dst->op_params + 0, sizeof(float));
                        memcpy(&max_bias, (float *)dst->op_params + 1, sizeof(float));

#pragma message("TODO: add ggml_vk_soft_max() F16 src1 support")
#pragma message("ref:  https://github.com/ggerganov/llama.cpp/pull/5021")
                        GGML_ASSERT(!src1 || src1t == GGML_TYPE_F32);

#pragma message("TODO: add ALiBi support")
#pragma message("ref:  https://github.com/ggerganov/llama.cpp/pull/7192")
                        GGML_ASSERT(max_bias == 0.0f);

                        ggml_vk_soft_max(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, ne01, ne02, ne03, scale);
                    } break;
                case GGML_OP_DIAG_MASK_INF:
                    {
                        const int n_past = ((int32_t *)(dst->op_params))[0];
                        ggml_vk_diag_mask_inf(seq, id_src0, id_dst, off_src0, off_dst, n_past, ne00, ne01, ne02);
                    } break;
                case GGML_OP_NORM:
                    {
                        float eps;
                        memcpy(&eps, dst->op_params, sizeof(float));
                        ggml_vk_norm(seq, id_src0, id_dst, off_src0, off_dst, ne00, nb01, ggml_nrows(src0), eps);
                    } break;
                case GGML_OP_RMS_NORM:
                    {
                        GGML_ASSERT(ne00 % 4 == 0);

                        float eps;
                        memcpy(&eps, dst->op_params, sizeof(float));
                        ggml_vk_rms_norm(seq, id_src0, id_dst, off_src0, off_dst, ne00, nb01, ggml_nrows(src0), eps);
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        GGML_ASSERT(ne00 == ne10);

                        GGML_ASSERT(ne12 % ne02 == 0);
                        GGML_ASSERT(ne13 % ne03 == 0);

                        const uint32_t r2 = ne12/ne02;
                        const uint32_t r3 = ne13/ne03;

                        if (src1t != GGML_TYPE_F32) {
                            fprintf(stderr, "%s: %s: Unsupported src1 type: %u/%u\n", __func__, ggml_op_name(dst->op), src0t, src1t);
                            goto not_implemented;
                        }

                        if (ggml_is_transposed(src0) ||
                            ggml_is_transposed(src1)) {
                            fprintf(stderr, "%s: %s: matmul on tranposed tensor not supported: %u/%u\n", __func__, ggml_op_name(dst->op), src0t, src1t);
                            goto not_implemented;
                        }

                        switch (src0t) {
                            case GGML_TYPE_F32:
                                ggml_vk_mul_mat_mat_f32(
                                    seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                    ne00, ne01, ne02, nb01, nb02, ne11, ne12, nb11, nb12, nb1, nb2
                                );
                                break;
                            case GGML_TYPE_F16:
                                ggml_vk_mul_mat_f16(
                                    seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                    ne00, ne01, ne02, nb00, nb01, nb02, ne10, ne11, ne12, ne13, nb10, nb11, nb12,
                                    ne0, ne1, r2, r3
                                );
                                break;
                            case GGML_TYPE_Q8_0:
                                ggml_vk_mul_mat_q8_0(
                                    seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                    ne00, ne01, ne02, ne10, ne11, ne12, ne13, ne0, ne1, r2, r3
                                );
                                break;
                            case GGML_TYPE_Q4_0:
                                ggml_vk_mul_mat_q4_0(
                                    seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                    ne00, ne01, ne02, ne10, ne11, ne12, ne13, ne0, ne1, r2, r3
                                );
                                break;
                            case GGML_TYPE_Q4_1:
                                ggml_vk_mul_mat_q4_1(
                                    seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                    ne00, ne01, ne02, ne10, ne11, ne12, ne13, ne0, ne1, r2, r3
                                );
                                break;
                            case GGML_TYPE_Q6_K:
                                ggml_vk_mul_mat_q6_k(
                                    seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst,
                                    ne00, ne10, ne0, ne1, ne01, ne11, ne12, ne02
                                );
                                break;
                            default: {
                                fprintf(stderr, "%s: %s: Unsupported quantization: %u/%u\n", __func__, ggml_op_name(dst->op), src0t, src1t);
                                goto not_implemented;
                            }
                        }

                    } break;
                case GGML_OP_GET_ROWS:
                    {
                        if (src0t == GGML_TYPE_F32) {
                            ggml_vk_get_rows_f32(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_F16) {
                            ggml_vk_get_rows_f16(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_Q4_0) {
                            ggml_vk_get_rows_q4_0(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_Q4_1) {
                            ggml_vk_get_rows_q4_1(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else if (src0t == GGML_TYPE_Q6_K) {
                            ggml_vk_get_rows_q6_k(seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, ne00, nb01, nb1, ggml_nelements(src1));
                        } else {
                            fprintf(stderr, "%s: %s: Unsupported quantization: %u\n", __func__, ggml_op_name(dst->op), src0t);
                            goto not_implemented;
                        }
                    } break;
                case GGML_OP_ROPE:
                    {
#pragma message("TODO: implement phi3 frequency factors support")
#pragma message("      https://github.com/ggerganov/llama.cpp/pull/7225")
                        GGML_ASSERT(dst->src[2] == nullptr && "phi3 frequency factors not implemented yet");

#pragma message("TODO: update rope NORM mode to match NEOX mode")
#pragma message("      https://github.com/ggerganov/llama.cpp/pull/7634")

                        GGML_ASSERT(ne10 == ne02);
                        GGML_ASSERT(src0t == dstt);
                        // const int n_past = ((int32_t *) dst->op_params)[0];
                        const int n_dims     = ((int32_t *) dst->op_params)[1];
                        const int mode       = ((int32_t *) dst->op_params)[2];
                        // skip 3, n_ctx used in GLM RoPE, unimplemented in Vulkan
                        const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

                        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
                        memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
                        memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
                        memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
                        memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
                        memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
                        memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
                        ggml_vk_rope(
                            seq, id_src0, id_src1, id_dst, off_src0, off_src1, off_dst, src0t, n_dims, mode, n_ctx_orig,
                            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
                            ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, nb0, nb1, nb2, nb3
                        );
                    } break;
                case GGML_OP_DUP:
                case GGML_OP_CPY:
                case GGML_OP_CONT:
                    {
                        switch (src0t) {
                            case GGML_TYPE_F32:
                                {
                                    switch (dstt) {
                                        case GGML_TYPE_F16: ggml_vk_cpy_f32_f16(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                        case GGML_TYPE_F32: ggml_vk_cpy_f32_f32(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                        default: goto not_implemented;
                                    }
                                } break;
                            case GGML_TYPE_F16:
                                {
                                    switch (dstt) {
                                        case GGML_TYPE_F16: ggml_vk_cpy_f16_f16(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                        case GGML_TYPE_F32: ggml_vk_cpy_f16_f32(seq, id_src0, id_dst, off_src0, off_dst, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne0, ne1, ne2, nb0, nb1, nb2, nb3); break;
                                    default: goto not_implemented;
                                } break;
                            default: goto not_implemented;
                            }
                        }
                    } break;
                default: goto not_implemented;
            }
            continue;
            not_implemented: {}
            fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
            //GGML_ABORT("fatal error");
        }

        // Evaluate sequence
        if (any_commands_recorded) {
            seq.evalAsync();
        }
    }

    // Wait for all sequences to finish
    for (auto& sequence : sequences) {
        if (sequence->isRunning())
            sequence->evalAwait();
    }

    ggml_vk_free_descriptor_pool(ctx);
}

template<>
kp::Tensor::TensorDataTypes
kp::TensorT<half>::dataType()
{
    return TensorDataTypes::eFloat;
}

template<>
kp::Tensor::TensorDataTypes
kp::TensorT<uint8_t>::dataType()
{
    return TensorDataTypes::eUnsignedInt;
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

struct ggml_backend_kompute_buffer_type_context {
    int         device;
    int         device_ref = 0;
    uint64_t    buffer_alignment;
    uint64_t    max_alloc;
    std::string name;

    ggml_backend_kompute_buffer_type_context(int device, uint64_t buffer_alignment, uint64_t max_alloc)
        : device(device), buffer_alignment(buffer_alignment), max_alloc(max_alloc), name(ggml_kompute_format_name(device)) {}
};

static void ggml_backend_kompute_device_ref(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_kompute_buffer_type_context *>(buft->context);

    if (!ctx->device_ref) {
        komputeManager()->initializeDevice(
            ctx->device, {}, {
                "VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage",
                "VK_KHR_16bit_storage", "VK_KHR_shader_non_semantic_info"
            }
        );
    }

    assert(ggml_vk_has_device());
    ctx->device_ref++;
}

static void ggml_backend_kompute_device_unref(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_kompute_buffer_type_context *>(buft->context);

    assert(ctx->device_ref > 0);

    ctx->device_ref--;

    if (!ctx->device_ref) {
        komputeManager.destroy();
    }
}

static const char * ggml_backend_kompute_buffer_get_name(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<ggml_backend_kompute_buffer_type_context *>(buffer->buft->context);
    return ctx->name.c_str();
}

static void ggml_backend_kompute_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * memory = (ggml_vk_memory *)buffer->context;
    if (ggml_vk_has_device()) {
        ggml_vk_free_memory(*memory);
    }
    delete memory;
}

static void * ggml_backend_kompute_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((ggml_vk_memory *)buffer->context)->data;
}

static void ggml_backend_kompute_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    const auto res = ggml_vk_get_tensor(tensor);
    GGML_ASSERT(res);

    memcpy((char *)tensor->data + offset, data, size);

    komputeManager()->sequence()->eval<kp::OpTensorSyncDevice>({res});
}

static void ggml_backend_kompute_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);

    const auto res = ggml_vk_get_tensor(tensor);
    GGML_ASSERT(res);

    komputeManager()->sequence()->eval<kp::OpTensorSyncLocal>({res});

    memcpy(data, (const char *)tensor->data + offset, size);
}

static void ggml_backend_kompute_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * memory = (ggml_vk_memory *)buffer->context;
    memset(memory->data, value, buffer->size);

    if (memory->stagingBuffer)
        komputeManager()->sequence()->eval<kp::OpBufferSyncDevice>(memory->primaryBuffer, memory->stagingBuffer, memory->size);
}

static ggml_backend_buffer_i ggml_backend_kompute_buffer_i = {
    /* .get_name        = */ ggml_backend_kompute_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_kompute_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_kompute_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .set_tensor      = */ ggml_backend_kompute_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_kompute_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_kompute_buffer_clear,
    /* .reset           = */ NULL,
};

// default buffer type

static const char * ggml_backend_kompute_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_kompute_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_kompute_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_kompute_device_ref(buft);
    auto * ctx = new ggml_vk_memory(ggml_vk_allocate(size));
    return ggml_backend_buffer_init(buft, ggml_backend_kompute_buffer_i, ctx, size);
}

static size_t ggml_backend_kompute_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_kompute_buffer_type_context *>(buft->context);
    return ctx->buffer_alignment;
}

static size_t ggml_backend_vk_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_kompute_buffer_type_context *>(buft->context);
    return ctx->max_alloc;
}

static ggml_backend_buffer_type_i ggml_backend_kompute_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_kompute_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_kompute_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_kompute_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_kompute_buffer_type(int device) {
    static std::vector<ggml_backend_buffer_type> bufts = []() {
        std::vector<ggml_backend_buffer_type> vec;
        auto devices = ggml_vk_available_devices_internal(0);
        vec.reserve(devices.size());

        for (const auto & dev : devices) {
            vec.push_back({
                /* .iface   = */ ggml_backend_kompute_buffer_type_interface,
                /* .context = */ new ggml_backend_kompute_buffer_type_context(dev.index, dev.bufferAlignment, dev.maxAlloc)
            });
        }
        return vec;
    }();

    auto it = std::find_if(bufts.begin(), bufts.end(), [device](const ggml_backend_buffer_type & t) {
        return device == static_cast<ggml_backend_kompute_buffer_type_context *>(t.context)->device;
    });
    return it < bufts.end() ? &*it : nullptr;
}

// backend

static const char * ggml_backend_kompute_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_kompute_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_kompute_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_kompute_context *>(backend->context);

    assert(ctx == s_kompute_context);
    s_kompute_context = nullptr;
    if (ctx != nullptr) {
        delete ctx;
    }

    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_kompute_get_default_buffer_type(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_kompute_context *>(backend->context);
    return ggml_backend_kompute_buffer_type(ctx->device);
}

static ggml_status ggml_backend_kompute_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_kompute_context *>(backend->context);
    ggml_vk_graph_compute(ctx, cgraph);
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_kompute_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_UNUSED(backend);
    return ggml_vk_supports_op(op);
}

static bool ggml_backend_kompute_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(backend);
    return buft->iface.get_name == ggml_backend_kompute_buffer_type_get_name;
}

static struct ggml_backend_i kompute_backend_i = {
    /* .get_name                = */ ggml_backend_kompute_name,
    /* .free                    = */ ggml_backend_kompute_free,
    /* .get_default_buffer_type = */ ggml_backend_kompute_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_kompute_graph_compute,
    /* .supports_op             = */ ggml_backend_kompute_supports_op,
    /* .supports_buft           = */ ggml_backend_kompute_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_kompute_guid() {
    static ggml_guid guid = { 0x7b, 0x57, 0xdc, 0xaf, 0xde, 0x12, 0x1d, 0x49, 0xfb, 0x35, 0xfa, 0x9b, 0x18, 0x31, 0x1d, 0xca };
    return &guid;
}

ggml_backend_t ggml_backend_kompute_init(int device) {
    GGML_ASSERT(s_kompute_context == nullptr);
    s_kompute_context = new ggml_kompute_context(device);

    ggml_backend_t kompute_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_kompute_guid(),
        /* .interface = */ kompute_backend_i,
        /* .context   = */ s_kompute_context,
    };

    return kompute_backend;
}

bool ggml_backend_is_kompute(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_kompute_guid());
}

static ggml_backend_t ggml_backend_reg_kompute_init(const char * params, void * user_data) {
    GGML_UNUSED(params);
    return ggml_backend_kompute_init(intptr_t(user_data));
}

extern "C" int ggml_backend_kompute_reg_devices();

int ggml_backend_kompute_reg_devices() {
    auto devices = ggml_vk_available_devices_internal(0);
    for (const auto & device : devices) {
        ggml_backend_register(
            ggml_kompute_format_name(device.index).c_str(),
            ggml_backend_reg_kompute_init,
            ggml_backend_kompute_buffer_type(device.index),
            reinterpret_cast<void *>(intptr_t(device.index))
        );
    }
    return devices.size();
}
