#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_greedy(void);

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_temp(float temp);

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_top_k(int32_t k);

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_dist(uint32_t seed);

#ifdef __cplusplus
}
#endif
