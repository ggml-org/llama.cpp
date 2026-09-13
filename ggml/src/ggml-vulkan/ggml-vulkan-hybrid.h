#pragma once

struct ggml_tensor;

bool ggml_vk_hybrid_supported(const ggml_tensor * node);
bool ggml_vk_hybrid_try(ggml_tensor * node);
