#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration

#ifdef GGML_HIP_GFX906

// ============================================
// MMQ Kernel Configuration
// ============================================
#define GFX906_MMQ_NWARPS 2

// ============================================
// Q8 Cache Configuration
// ============================================
#define GFX906_KVQ_MOE_CACHE_ENABLED 0
// Layer-cycling: N cycles, slot size = TOTAL / N
#define GFX906_Q8_CACHE_TOTAL_SIZE      (128 * 1024 * 1024)  // Total cache size: 128MB
#define GFX906_Q8_CACHE_NUM_SLOTS       1                    // Number of cycles  
#define GFX906_Q8_CACHE_LAYERS_PER_SLOT 1                    // 1 layer per slot

// ============================================
// ROPE Optimization
// ============================================
#define GFX906_ROPE_ENABLED 1

#endif // GGML_HIP_GFX906
