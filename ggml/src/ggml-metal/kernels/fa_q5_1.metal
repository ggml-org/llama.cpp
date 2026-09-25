#include "common.h"
#include "dequantize.h"
#include "fa_common.metal"

#define FA_TYPES \
    half,   half4,     simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    half,   half4x4,   simdgroup_half8x8,  \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

#define FA_TYPES_BF \
    bfloat, bfloat4,   simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    bfloat, bfloat4x4, simdgroup_bfloat8x8, \
    float,             simdgroup_float8x8,  \
    float,  float2,    simdgroup_float8x8,  \
    half,   half4,     simdgroup_half8x8
    //float,  float4,    simdgroup_float8x8

#define FA_TYPES_F32 \
    half,   half4,     simdgroup_half8x8,  \
    float,  float4x4,  simdgroup_float8x8, \
    float,  float4x4,  simdgroup_float8x8, \
    float,             simdgroup_float8x8, \
    float,  float2,    simdgroup_float8x8, \
    float,  float4,    simdgroup_float8x8
    //half,   half4,     simdgroup_half8x8

typedef decltype(kernel_flash_attn_ext<FA_TYPES, half4x4, 1, dequantize_f16, half4x4, 1, dequantize_f16, 64, 64>) flash_attn_ext_t;

template [[host_name("kernel_flash_attn_ext_q5_1_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 32,  32>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk40_dv40"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 40,  40>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk48_dv48"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 48,  48>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 64,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk72_dv72"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 72,  72>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk80_dv80"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 80,  80>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  96>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk96_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 96,  64>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk112_dv112")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 112, 112>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 128, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 192>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 192, 128>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 256, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 320, 256>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 512, 512>;
template [[host_name("kernel_flash_attn_ext_q5_1_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES,    block_q5_1, 2, dequantize_q5_1, block_q5_1, 2, dequantize_q5_1, 576, 512>;

#undef FA_TYPES
#undef FA_TYPES_BF
#undef FA_TYPES_F32
