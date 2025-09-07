// Placeholder Vulkan GPU performance counter collection API.
// Currently no-ops; wired in ggml-vulkan backend for future enablement.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to command buffer (as void* to avoid including Vulkan headers here).
// `label` is a short name for the region, e.g., pipeline name.
void llama_vk_counters_begin(void * command_buffer, const char * label);
void llama_vk_counters_end(void * command_buffer);

#ifdef __cplusplus
}
#endif

