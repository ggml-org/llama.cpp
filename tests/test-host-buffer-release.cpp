// ref: https://github.com/ggml-org/llama.cpp/issues/30556
// only iOS and Mac Catalyst builds fail without the fix, macOS already uses vm_allocate
// thus, this test is not run by default

#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mach/mach.h>

static uint64_t phys_footprint() {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    const kern_return_t ret = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count);
    GGML_ASSERT(ret == KERN_SUCCESS);
    return info.phys_footprint;
}

static bool test_release(ggml_backend_buffer_type_t buft) {
    const size_t size = 256ull*1024*1024;

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, size);
    GGML_ASSERT(buf != nullptr);
    ggml_backend_buffer_clear(buf, 1);

    const uint64_t footprint_alloc = phys_footprint();
    ggml_backend_buffer_free(buf);
    const uint64_t footprint_free = phys_footprint();

    printf("%s: phys_footprint %.1f MiB -> %.1f MiB after freeing %.1f MiB\n", ggml_backend_buft_name(buft), footprint_alloc/1048576.0, footprint_free/1048576.0, size/1048576.0);

    // do not poll, the footprint must drop at free time
    return footprint_free < footprint_alloc - 0.75*size;
}

int main() {
    ggml_backend_load_all();

    bool ok = test_release(ggml_backend_cpu_buffer_type());

    // private Metal buffers are allocated by Metal, not by ggml_metal_host_malloc
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev && strcmp(ggml_backend_buft_name(ggml_backend_dev_buffer_type(dev)), "MTL0") == 0) {
        ok = test_release(ggml_backend_dev_buffer_type(dev)) && ok;
    }

    return ok ? 0 : 1;
}
