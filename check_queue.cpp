#include <level_zero/ze_api.h>
#include <cstdio>
#include <vector>

int main() {
    if (zeInit(0) != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to initialize Level Zero\n");
        return 1;
    }
    ze_driver_handle_t driver = nullptr;
    uint32_t count = 1;
    if (zeDriverGet(&count, &driver) != ZE_RESULT_SUCCESS || count == 0) {
        fprintf(stderr, "Failed to find Level Zero driver\n");
        return 1;
    }
    ze_device_handle_t device = nullptr;
    count = 1;
    if (zeDeviceGet(driver, &count, &device) != ZE_RESULT_SUCCESS || count == 0) {
        fprintf(stderr, "Failed to find Level Zero device\n");
        return 1;
    }

    uint32_t group_count = 0;
    if (zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr) != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to count command queue groups\n");
        return 1;
    }
    if (group_count == 0) {
        return 0;
    }
    std::vector<ze_command_queue_group_properties_t> props(group_count);
    for (auto & prop : props) {
        prop.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        prop.pNext = nullptr;
    }
    if (zeDeviceGetCommandQueueGroupProperties(device, &group_count, props.data()) != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "Failed to query command queue groups\n");
        return 1;
    }

    for (uint32_t i = 0; i < group_count; ++i) {
        printf("Group %u: flags=%u\n", i, props[i].flags);
        if ((props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            !(props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
            printf("Group %u is Copy Engine\n", i);
        }
    }
    return 0;
}
