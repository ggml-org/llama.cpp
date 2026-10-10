// CPU-only: c++ -std=c++17 tests/test-check-queue.cpp -o /tmp/test-check-queue
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>

static int copy_engine_reports;
static int probe_printf(const char * format, ...);

#define main check_queue_main
#define printf probe_printf
#include "../check_queue.cpp"
#undef printf
#undef main

static int scenario;
static int calls;

static int probe_printf(const char * format, ...) {
    va_list args;
    va_start(args, format);
    if (strcmp(format, "Group %u is Copy Engine\n") == 0) {
        // A COMPUTE|COPY group is not a dedicated copy engine.
        assert(va_arg(args, unsigned int) == 0);
        ++copy_engine_reports;
    }
    va_end(args);
    return 0;
}

ze_result_t ZE_APICALL zeInit(ze_init_flags_t) {
    ++calls;
    return scenario == 1 ? ZE_RESULT_ERROR_UNINITIALIZED : ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL zeDriverGet(uint32_t * count, ze_driver_handle_t * driver) {
    ++calls;
    assert(*count == 1 && *driver == nullptr);
    if (scenario == 2) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    *count = scenario == 3 ? 0 : 1;
    *driver = reinterpret_cast<ze_driver_handle_t>(1);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL zeDeviceGet(ze_driver_handle_t driver, uint32_t * count, ze_device_handle_t * device) {
    ++calls;
    assert(driver != nullptr && *count == 1 && *device == nullptr);
    if (scenario == 4) {
        return ZE_RESULT_ERROR_UNINITIALIZED;
    }
    *count = scenario == 5 ? 0 : 1;
    *device = reinterpret_cast<ze_device_handle_t>(1);
    return ZE_RESULT_SUCCESS;
}

ze_result_t ZE_APICALL zeDeviceGetCommandQueueGroupProperties(
        ze_device_handle_t device, uint32_t * count, ze_command_queue_group_properties_t * props) {
    ++calls;
    assert(device != nullptr);
    if (!props) {
        assert(*count == 0);
        *count = scenario == 7 ? 0 : 2;
        return scenario == 6 ? ZE_RESULT_ERROR_UNINITIALIZED : ZE_RESULT_SUCCESS;
    }
    assert(*count == 2);
    for (uint32_t i = 0; i < *count; ++i) {
        assert(props[i].stype == ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES);
        assert(props[i].pNext == nullptr);
        props[i].flags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        if (i == 1) {
            props[i].flags |= ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
        }
    }
    return scenario == 8 ? ZE_RESULT_ERROR_UNINITIALIZED : ZE_RESULT_SUCCESS;
}

int main() {
    const int expected_calls[] = {5, 1, 2, 2, 3, 3, 4, 4, 5};
    for (scenario = 0; scenario < 9; ++scenario) {
        calls = 0;
        copy_engine_reports = 0;
        const int result = check_queue_main();
        assert(result == (scenario == 0 || scenario == 7 ? 0 : 1));
        assert(calls == expected_calls[scenario]);
        assert(copy_engine_reports == (scenario == 0 ? 1 : 0));
    }
    puts("9 queue-probe scenarios passed");
}
