#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration

#ifdef GGML_USE_HIP
#define GFX906_MMQ_ITER_K 256
#define GFX906_MMQ_NWARPS 2
#define GFX906_FATTN_Q8_ENABLED 1
#define GFX906_Q8_SUPPORTS_HEAD_DIM(d) \
    ((d) % 32 == 0 && (d) != 40 && (d) != 80 && (d) != 112)

#define GFX906_USE_DPP_REDUCTIONS 1
#define GFX906_FATTN_TILE_SIZE_DEFAULT 128
#define GFX906_Q8_SCALE_HOISTING 1
#define GFX906_KVQ_MOE_CACHE_ENABLED 0
//cache disabled cause it has problems with different models. 
//still needs to be debugged
#define GFX906_ROPE_ENABLED 1

#endif // GGML_USE_HIP
