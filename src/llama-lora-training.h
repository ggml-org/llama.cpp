#pragma once

#include "llama.h"
#include "llama-model.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "ggml.h"

#include <vector>

bool llama_lora_validate_training_params(const llama_adapter_lora_train_params * params);

ggml_context * llama_lora_create_context(size_t mem_size);

void llama_lora_create_tensor_pair(
    ggml_context * lora_ctx,
    const char * base_name,
    const ggml_tensor * base_tensor,
    int32_t rank,
    ggml_tensor ** lora_a,
    ggml_tensor ** lora_b);

llama_adapter_lora * llama_lora_create_adapter(
    llama_model * model,
    const llama_adapter_lora_train_params * params);

bool llama_lora_allocate_buffers(
    llama_adapter_lora * adapter,
    const std::vector<ggml_backend_buffer_type_t> & bufts);
