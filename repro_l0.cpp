#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define ZE_CALL(call) do { \
    ze_result_t res = (call); \
    if (res != ZE_RESULT_SUCCESS) { \
        printf("ZE Call failed at line %d: %d\n", __LINE__, res); \
        exit(1); \
    } \
} while(0)

int main() {
    ze_driver_handle_t driver = nullptr;
    ze_device_handle_t device = nullptr;
    ze_context_handle_t context;
    ze_command_queue_handle_t queue;
    ze_command_list_handle_t cmd_list;

    ZE_CALL(zeInit(0));

    uint32_t driver_count = 1;
    ZE_CALL(zeDriverGet(&driver_count, &driver));
    if (driver_count == 0) {
        fprintf(stderr, "No Level Zero driver found\n");
        return 1;
    }

    uint32_t device_count = 1;
    ZE_CALL(zeDeviceGet(driver, &device_count, &device));
    if (device_count == 0) {
        fprintf(stderr, "No Level Zero device found\n");
        return 1;
    }

    uint32_t group_count = 0;
    ZE_CALL(zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr));
    std::vector<ze_command_queue_group_properties_t> groups(group_count);
    for (auto & group : groups) {
        group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }
    if (group_count != 0) {
        ZE_CALL(zeDeviceGetCommandQueueGroupProperties(device, &group_count, groups.data()));
    }
    uint32_t ordinal = 0;
    while (ordinal < group_count &&
           (!(groups[ordinal].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) ||
            groups[ordinal].numQueues == 0)) {
        ++ordinal;
    }
    if (ordinal == group_count) {
        fprintf(stderr, "No compute queue group found\n");
        return 1;
    }

    ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    ZE_CALL(zeContextCreate(driver, &ctx_desc, &context));

    ze_command_queue_desc_t queue_desc = {
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,
        ordinal,
        0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    ZE_CALL(zeCommandQueueCreate(context, device, &queue_desc, &queue));

    ze_command_list_desc_t cl_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, ordinal, 0};
    ZE_CALL(zeCommandListCreate(context, device, &cl_desc, &cmd_list));

    size_t copy_size = 860672;
    void* host_ptr = aligned_alloc(4096, copy_size);
    void* dev_ptr;
    ze_device_mem_alloc_desc_t alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr, 0, 0};
    ZE_CALL(zeMemAllocDevice(context, &alloc_desc, copy_size, 4096, device, &dev_ptr));

    for (int i = 0; i < 1000; ++i) {
        ZE_CALL(zeCommandListAppendMemoryCopy(cmd_list, dev_ptr, host_ptr, copy_size, nullptr, 0, nullptr));
        ZE_CALL(zeCommandListAppendBarrier(cmd_list, nullptr, 0, nullptr));
    }
    ZE_CALL(zeCommandListClose(cmd_list));
    ZE_CALL(zeCommandQueueExecuteCommandLists(queue, 1, &cmd_list, nullptr));
    constexpr uint64_t sync_timeout_ns = 30ULL * 1000 * 1000 * 1000;
    ZE_CALL(zeCommandQueueSynchronize(queue, sync_timeout_ns));

    printf("Done.\n");
    return 0;
}
