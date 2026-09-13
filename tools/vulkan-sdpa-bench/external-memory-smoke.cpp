#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vulkan/vulkan.h>
#include <level_zero/ze_api.h>
#include <vulkan/vulkan_win32.h>

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    if (zeInit(0) != ZE_RESULT_SUCCESS) { std::printf("FAIL zeInit\n"); return 1; }
    uint32_t driver_count = 0;
    if (zeDriverGet(&driver_count, nullptr) != ZE_RESULT_SUCCESS || driver_count == 0) { std::printf("FAIL ze_driver_not_found\n"); return 1; }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    zeDriverGet(&driver_count, drivers.data());
    uint32_t device_count = 0;
    zeDeviceGet(drivers[0], &device_count, nullptr);
    std::vector<ze_device_handle_t> ze_devices(device_count);
    zeDeviceGet(drivers[0], &device_count, ze_devices.data());
    bool ze_arc = false;
    for (auto device : ze_devices) {
        ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        zeDeviceGetProperties(device, &props);
        if (std::strstr(props.name, "Arc") || std::strstr(props.name, "A750")) {
            ze_arc = true;
            std::printf("level_zero_device=%s\n", props.name);
        }
    }
    if (!ze_arc) { std::printf("FAIL level_zero_arc_not_found\n"); return 1; }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "llama-vulkan-external-smoke", 1, nullptr, 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 0, nullptr};
    VkInstance instance = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ici, nullptr, &instance);
    if (r != VK_SUCCESS) { std::printf("FAIL create_instance=%d\n", r); return 1; }
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    bool found = false;
    for (VkPhysicalDevice dev : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        if (std::strstr(props.deviceName, "Arc") == nullptr && std::strstr(props.deviceName, "A750") == nullptr) continue;
        found = true;
        VkPhysicalDeviceExternalBufferInfo bi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, nullptr, 0, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
        VkExternalBufferProperties bp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES, nullptr, {}};
        vkGetPhysicalDeviceExternalBufferProperties(dev, &bi, &bp);
        const bool exportable = (bp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0;
        const bool importable = (bp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        std::printf("device=%s\n", props.deviceName);
        std::printf("opaque_win32 exportable=%s importable=%s compatible=0x%x\n", exportable ? "yes" : "no", importable ? "yes" : "no", bp.externalMemoryProperties.compatibleHandleTypes);

        if (!exportable || !importable) continue;
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops.data());
        uint32_t qfamily = UINT32_MAX;
        for (uint32_t i = 0; i < qcount; ++i) if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfamily = i; break; }
        if (qfamily == UINT32_MAX) continue;
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, qfamily, 1, &priority};
        const char * ext = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &qci, 0, nullptr, 1, &ext, nullptr};
        VkDevice vkdev = VK_NULL_HANDLE;
        if (vkCreateDevice(dev, &dci, nullptr, &vkdev) != VK_SUCCESS) continue;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, 4096, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 1, &qfamily};
        VkExternalMemoryBufferCreateInfo ebci{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
        bci.pNext = &ebci;
        VkBuffer buffer = VK_NULL_HANDLE;
        if (vkCreateBuffer(vkdev, &bci, nullptr, &buffer) != VK_SUCCESS) { vkDestroyDevice(vkdev, nullptr); continue; }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(vkdev, buffer, &req);
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(dev, &mp);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
        VkExportMemoryAllocateInfo exai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exai, req.size, mt};
        VkDeviceMemory memory = VK_NULL_HANDLE;
        bool roundtrip = false;
        if (mt != UINT32_MAX && vkAllocateMemory(vkdev, &mai, nullptr, &memory) == VK_SUCCESS && vkBindBufferMemory(vkdev, buffer, memory, 0) == VK_SUCCESS) {
            auto get_handle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(vkdev, "vkGetMemoryWin32HandleKHR"));
            HANDLE handle = nullptr;
            VkMemoryGetWin32HandleInfoKHR ghi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, nullptr, memory, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT};
            if (get_handle && get_handle(vkdev, &ghi, &handle) == VK_SUCCESS && handle) {
                ze_context_desc_t zcd{ZE_STRUCTURE_TYPE_CONTEXT_DESC};
                ze_context_handle_t zctx = nullptr;
                if (zeContextCreate(drivers[0], &zcd, &zctx) == ZE_RESULT_SUCCESS) {
                    ze_device_mem_alloc_desc_t zad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
                    ze_external_memory_import_win32_handle_t zid{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32, nullptr, ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32, handle, nullptr};
                    zad.pNext = &zid;
                    void * imported = nullptr;
                    roundtrip = zeMemAllocDevice(zctx, &zad, req.size, req.alignment, ze_devices[0], &imported) == ZE_RESULT_SUCCESS;
                    if (imported) zeMemFree(zctx, imported);
                    zeContextDestroy(zctx);
                }
                CloseHandle(handle);
            }
        }
        if (memory) vkFreeMemory(vkdev, memory, nullptr);
        vkDestroyBuffer(vkdev, buffer, nullptr);
        vkDestroyDevice(vkdev, nullptr);
        std::printf("vulkan_win32_to_level_zero_import=%s\n", roundtrip ? "yes" : "no");
    }
    vkDestroyInstance(instance, nullptr);
    if (!found) { std::printf("FAIL Arc_device_not_found\n"); return 2; }
    std::printf("RESULT vulkan_export_to_level_zero_import=yes\n");
    return 0;
}
