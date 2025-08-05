#pragma once

#include "llama.h"
#include "llama-model.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "ggml.h"


bool llama_lora_validate_training_params(const struct llama_lora_training_params * params);

ggml_context * llama_lora_create_context(size_t mem_size);

bool llama_lora_create_tensor_pair(
    struct ggml_context * lora_ctx,
    const char * base_name,
    const struct ggml_tensor * base_tensor,
    int32_t rank,
    struct ggml_tensor ** lora_a,
    struct ggml_tensor ** lora_b);

void llama_lora_init_tensor_weights(
    struct ggml_tensor * lora_a, 
    struct ggml_tensor * lora_b, 
    float init_std);

struct llama_adapter_lora * llama_lora_create_adapter(
    struct llama_model * model, 
    const struct llama_lora_training_params * params);

bool llama_lora_allocate_buffers(
    struct llama_adapter_lora * adapter, 
    struct llama_model * model);

