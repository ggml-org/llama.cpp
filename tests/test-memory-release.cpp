// ref: https://github.com/ggml-org/llama.cpp/issues/25937

#include <cstdint>
#include <unistd.h>
#include <mach/mach.h>

#include "llama.h"
#include "get-model.h"

constexpr uint64_t TOLERANCE = 50'000'000; // 50MB

static uint64_t memory_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &n) != KERN_SUCCESS) {
        return UINT64_MAX;
    }
    return info.phys_footprint;
}

int main(int argc, char ** argv) {
    auto * model_path = get_model_or_exit(argc, argv);

    const uint64_t footprint_initial = memory_footprint();
    const uint64_t footprint_threshold = footprint_initial + TOLERANCE;

    llama_backend_init();
    struct llama_model* model = llama_model_load_from_file(model_path, llama_model_default_params());

    GGML_ASSERT(memory_footprint() > footprint_threshold);

    llama_model_free(model);
    llama_backend_free();

    const int64_t t_start_ms = ggml_time_ms();

    while (memory_footprint() > footprint_threshold) {
        // expect memory usage to drop within 10 seconds
        GGML_ASSERT(ggml_time_ms() - t_start_ms < 10'000);
        usleep(100'000); // 100ms
    }

    return 0;
}
