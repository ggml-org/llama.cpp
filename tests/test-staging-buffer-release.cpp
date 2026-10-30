// checks that freeing a model loaded without mmap returns the process footprint to where it was
// needs a model file and the Metal backend, thus this test is not run by default
// a leaked staging buffer is as big as the largest offloaded tensor, so the model needs one well over 5% of the load
// example model to run with: LiquidAI/LFM2.5-350M-GGUF (Q4_K_M)

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <mach/mach.h>

static double phys_footprint_mib() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t ret = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count);
    GGML_ASSERT(ret == KERN_SUCCESS);
    return info.phys_footprint/1048576.0;
}

int main(int argc, char ** argv) {
    auto * model_path = common_get_model_or_exit(argc, argv);

    // private Metal buffers leave the footprint some time after the free, so use shared ones
    setenv("GGML_METAL_SHARED_BUFFERS_ENABLE", "1", 1);

    llama_backend_init();

    // the first load makes one-time allocations; use mmap for it, so it leaves no freed staging buffer for the measured load to reuse
    llama_model_params params = llama_model_default_params();
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);
    llama_model_free(model);

    const double footprint_init = phys_footprint_mib();

    params.load_mode = LLAMA_LOAD_MODE_NONE;
    model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    const double footprint_load = phys_footprint_mib();
    llama_model_free(model);
    const double footprint_free = phys_footprint_mib();

    printf("phys_footprint %.1f MiB -> %.1f MiB after load -> %.1f MiB after free\n", footprint_init, footprint_load, footprint_free);

    // freed malloc blocks can stay in the footprint for a while, so a staging buffer from malloc is still there right after free
    GGML_ASSERT(footprint_free - footprint_init < 0.05*(footprint_load - footprint_init));

    llama_backend_free();

    return 0;
}
