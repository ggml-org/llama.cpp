#pragma once

#include <cstdint>

namespace turboquant {

// KV Cache block eviction parameters
constexpr uint32_t DEFAULT_BLOCK_SIZE = 32;
constexpr uint32_t MAX_BLOCK_SIZE = 64;
constexpr uint32_t MIN_BLOCK_SIZE = 16;

// TurboQuant quantization parameters
constexpr int DEFAULT_BIT_WIDTH = 4;  // 4-bit quantization
constexpr int QJL_BIT_WIDTH = 1;      // 1-bit QJL transform
constexpr int DEFAULT_SEED = 42;      // Default random seed for reproducibility

// Attention sink parameters (StreamingLLM)
constexpr uint32_t DEFAULT_ATTENTION_SINK_COUNT = 4;
constexpr uint32_t MIN_ATTENTION_SINK_COUNT = 1;
constexpr uint32_t MAX_ATTENTION_SINK_COUNT = 16;

// H2O (Heavy Hitter Oracle) parameters
constexpr float DEFAULT_HEAVY_HITTER_RATIO = 0.2f;  // Keep top 20% as heavy hitters
constexpr float MIN_HEAVY_HITTER_RATIO = 0.1f;
constexpr float MAX_HEAVY_HITTER_RATIO = 0.5f;

// Local window parameters
constexpr uint32_t DEFAULT_LOCAL_WINDOW_SIZE = 2048;  // Keep last 2K tokens
constexpr uint32_t MIN_LOCAL_WINDOW_SIZE = 256;
constexpr uint32_t MAX_LOCAL_WINDOW_SIZE = 16384;

// MiniCache layer merging threshold
constexpr float MINICACHE_SIMILARITY_THRESHOLD = 0.95f;

// Q-Filter geometry threshold
constexpr float QFILTER_GEOMETRY_THRESHOLD = 0.1f;

// AsymKV key homogeneity threshold
constexpr float ASYMKV_HOMOGENEITY_THRESHOLD = 0.9f;

// Memory management
constexpr size_t CACHE_ALIGNMENT = 64;  // Align cache to cache line size

}  // namespace turboquant
