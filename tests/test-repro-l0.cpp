// CPU-only: c++ -std=c++17 tests/test-repro-l0.cpp -o /tmp/test-repro-l0
#include <cassert>
#include <sys/wait.h>
#include <unistd.h>
#define main repro_main
#include "../repro_l0.cpp"
#undef main

static int scenario;
static uint32_t queue_ordinal, list_ordinal;
static bool submitted;
ze_result_t ZE_APICALL zeInit(ze_init_flags_t) { return ZE_RESULT_SUCCESS; }
ze_result_t ZE_APICALL zeDriverGet(uint32_t *, ze_driver_handle_t * h) {
    *h = reinterpret_cast<ze_driver_handle_t>(1); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeDeviceGet(ze_driver_handle_t, uint32_t *, ze_device_handle_t * h) {
    *h = reinterpret_cast<ze_device_handle_t>(1); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeDeviceGetCommandQueueGroupProperties(
        ze_device_handle_t, uint32_t * n, ze_command_queue_group_properties_t * p) {
    if (!p) { *n = scenario == 3 ? 0 : 3; return ZE_RESULT_SUCCESS; }
    for (uint32_t i = 0; i < *n; ++i) {
        assert(p[i].stype == ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES && !p[i].pNext);
        p[i].flags = i == (scenario == 0 ? 0u : 2u) && scenario != 2
            ? ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE : ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
        p[i].numQueues = 1;
    }
    return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeContextCreate(ze_driver_handle_t, const ze_context_desc_t *, ze_context_handle_t * h) {
    *h = reinterpret_cast<ze_context_handle_t>(1); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeCommandQueueCreate(ze_context_handle_t, ze_device_handle_t,
        const ze_command_queue_desc_t * d, ze_command_queue_handle_t * h) {
    queue_ordinal = d->ordinal; *h = reinterpret_cast<ze_command_queue_handle_t>(1); return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeCommandListCreate(ze_context_handle_t, ze_device_handle_t,
        const ze_command_list_desc_t * d, ze_command_list_handle_t * h) {
    list_ordinal = d->commandQueueGroupOrdinal; *h = reinterpret_cast<ze_command_list_handle_t>(1);
    return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeMemAllocDevice(ze_context_handle_t, const ze_device_mem_alloc_desc_t *,
        size_t, size_t, ze_device_handle_t, void ** p) { *p = nullptr; return ZE_RESULT_SUCCESS; }
ze_result_t ZE_APICALL zeCommandListAppendMemoryCopy(ze_command_list_handle_t, void *, const void *,
        size_t, ze_event_handle_t, uint32_t, ze_event_handle_t *) { return ZE_RESULT_SUCCESS; }
ze_result_t ZE_APICALL zeCommandListAppendBarrier(ze_command_list_handle_t, ze_event_handle_t,
        uint32_t, ze_event_handle_t *) { return ZE_RESULT_SUCCESS; }
ze_result_t ZE_APICALL zeCommandListClose(ze_command_list_handle_t) { return ZE_RESULT_SUCCESS; }
ze_result_t ZE_APICALL zeCommandQueueExecuteCommandLists(ze_command_queue_handle_t, uint32_t,
        ze_command_list_handle_t *, ze_fence_handle_t) {
    assert(queue_ordinal == list_ordinal);
    assert(queue_ordinal == (scenario == 0 ? 0u : 2u));
    submitted = true; return ZE_RESULT_SUCCESS;
}
ze_result_t ZE_APICALL zeCommandQueueSynchronize(ze_command_queue_handle_t, uint64_t timeout_ns) {
    // A stalled queue must have a finite deadline, not UINT64_MAX.
    assert(timeout_ns > 0 && timeout_ns <= 30ULL * 1000 * 1000 * 1000);
    return scenario == 4 ? ZE_RESULT_NOT_READY : ZE_RESULT_SUCCESS;
}
int main() {
    for (scenario = 0; scenario < 4; ++scenario) {
        submitted = false;
        assert(repro_main() == (scenario < 2 ? 0 : 1));
        assert(submitted == (scenario < 2));
    }
    fflush(nullptr);
    const pid_t child = fork();
    assert(child >= 0);
    if (child == 0) {
        scenario = 4;
        repro_main();
        _exit(0);
    }
    int status;
    assert(waitpid(child, &status, 0) == child);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    puts("4 queue-ordinal scenarios and synchronization timeout passed");
}
