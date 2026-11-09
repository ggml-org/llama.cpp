#include "common.h"
#include "dequantize.h"
#include "fa_vec_common.metal"

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

// baseline config (Q == 1): one instantiation per (dk,dv) for every simdgroup width.
// NE follows the width via FA_VEC_NE (see fa_vec_common.metal), so this list is not
// duplicated per width.
template [[host_name("kernel_flash_attn_ext_vec_f32_dk32_dv32")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  32, 32, FA_VEC_NE(32, 32)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk64_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  64, 64, FA_VEC_NE(64, 64)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 96, FA_VEC_NE(96, 96)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk96_dv64")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  96, 64, FA_VEC_NE(96, 64)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk128_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  128, 128, FA_VEC_NE(128, 128)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk128_dv96")]]    kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  128, 96, FA_VEC_NE(128, 96)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 192, FA_VEC_NE(192, 192)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 128, FA_VEC_NE(192, 128)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  256, 256, FA_VEC_NE(256, 256)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  320, 256, FA_VEC_NE(320, 256)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  512, 512, FA_VEC_NE(512, 512)>;
template [[host_name("kernel_flash_attn_ext_vec_f32_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  576, 512, FA_VEC_NE(576, 512)>;

#if N_SIMDWIDTH == 32
// tuned (Q,NE) variants: the tuned table and these explicit NE values are 32-lane only
#endif

#undef FA_TYPES
#undef FA_TYPES_F32
