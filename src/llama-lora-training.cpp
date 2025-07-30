#include "llama.h"
#include "ggml.h"
#include "llama-impl.h"

#include <cstring>


bool llama_opt_param_filter_lora(const struct ggml_tensor * tensor, void * userdata) {
    (void) userdata; // Unused param

    if (!tensor || !tensor->name) {
        return false;
    }

    const char * name = tensor->name;

    // Check if tensor is LoRA A or B
    // LoRA tensor naming convention: blk.{layer}.{module}.lora_a or .lora_b
    if (strstr(name, ".lora_a") || strstr(name, ".lora_b")) {
        LLAMA_LOG_DEBUG("LoRA filter: including trainable params '%s'\n", name);
        return true;
    }

    return false;
}