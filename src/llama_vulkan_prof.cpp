// Stub implementation of Vulkan GPU performance counter collection.
// Filled in later for platforms/extensions that support it.

#include <stddef.h>
#include "../include/llama_vulkan_prof.h"

extern "C" void llama_vk_counters_begin(void * command_buffer, const char * label) {
    (void)command_buffer;
    (void)label;
}

extern "C" void llama_vk_counters_end(void * command_buffer) {
    (void)command_buffer;
}

