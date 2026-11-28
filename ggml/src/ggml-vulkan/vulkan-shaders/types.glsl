#if !defined(GGML_TYPES_COMP)
#define GGML_TYPES_COMP

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_16bit_storage : require

#ifdef USE_OCP_FP4
#extension GL_EXT_float_e2m1 : require
#extension GL_EXT_float_e4m3 : require
#endif

#if defined(DATA_A_F32)
#define QUANT_K 1
#define QUANT_R 1

#if LOAD_VEC_A == 4
#define A_TYPE vec4
#elif LOAD_VEC_A == 8
#define A_TYPE mat2x4
#else
#define A_TYPE float
#endif
#define A_TYPE_PACKED64 vec2
#endif

#if defined(DATA_A_F16)
#define QUANT_K 1
#define QUANT_R 1

#if LOAD_VEC_A == 4
#define A_TYPE f16vec4
#elif LOAD_VEC_A == 8
#define A_TYPE f16mat2x4
#else
#define A_TYPE float16_t
#endif
#define A_TYPE_PACKED32 f16vec2
#endif

#if defined(DATA_A_BF16)
#define QUANT_K 1
#define QUANT_R 1

#if LOAD_VEC_A == 4
#define A_TYPE u16vec4
#elif LOAD_VEC_A == 8
#error unsupported
#else
#define A_TYPE uint16_t
#endif
#define A_TYPE_PACKED32 uint32_t
#endif

#define QUANT_K_Q4_0 32
#define QUANT_R_Q4_0 2

struct block_q4_0
{
    float16_t d;
    uint8_t qs[16];
};
struct block_q4_0_packed16
{
    float16_t d;
    uint16_t qs[16/2];
};

#if defined(DATA_A_Q4_0)
#define QUANT_K QUANT_K_Q4_0
#define QUANT_R QUANT_R_Q4_0
#define QUANT_AUXF 1
#define A_TYPE block_q4_0
#define A_TYPE_PACKED16 block_q4_0_packed16
#define DATA_A_QUANT_LEGACY
#endif

#define QUANT_K_Q4_1 32
#define QUANT_R_Q4_1 2

struct block_q4_1
{
    float16_t d;
    float16_t m;
    uint8_t qs[16];
};

struct block_q4_1_packed16
{
    float16_t d;
    float16_t m;
    uint16_t qs[16/2];
};

struct block_q4_1_packed32
{
    f16vec2 dm;
    uint32_t qs[16/4];
};

#if defined(DATA_A_Q4_1)
#define QUANT_K QUANT_K_Q4_1
#define QUANT_R QUANT_R_Q4_1
#define QUANT_AUXF 2
#define A_TYPE block_q4_1
#define A_TYPE_PACKED16 block_q4_1_packed16
#define A_TYPE_PACKED32 block_q4_1_packed32
#define DATA_A_QUANT_LEGACY
#endif

#define QUANT_K_Q5_0 32
#define QUANT_R_Q5_0 2

struct block_q5_0
{
    float16_t d;
    uint16_t qh[2];
    uint8_t qs[16];
};

struct block_q5_0_packed16
{
    float16_t d;
    uint16_t qh[2];
    uint16_t qs[16/2];
};

#if defined(DATA_A_Q5_0)
#define QUANT_K QUANT_K_Q5_0
#define QUANT_R QUANT_R_Q5_0
#define QUANT_AUXF 1
#define A_TYPE block_q5_0
#define A_TYPE_PACKED16 block_q5_0_packed16
#define DATA_A_QUANT_LEGACY
#endif

#define QUANT_K_Q5_1 32
#define QUANT_R_Q5_1 2

struct block_q5_1
{
    float16_t d;
    float16_t m;
    uint qh;
    uint8_t qs[16];
};

struct block_q5_1_packed16
{
    float16_t d;
    float16_t m;
    uint qh;
    uint16_t qs[16/2];
};

struct block_q5_1_packed32
{
    f16vec2 dm;
    uint qh;
    uint32_t qs[16/4];
};

#if defined(DATA_A_Q5_1)
#define QUANT_K QUANT_K_Q5_1
#define QUANT_R QUANT_R_Q5_1
#define QUANT_AUXF 2
#define A_TYPE block_q5_1
#define A_TYPE_PACKED16 block_q5_1_packed16
#define A_TYPE_PACKED32 block_q5_1_packed32
#define DATA_A_QUANT_LEGACY
#endif

#define QUANT_K_Q8_0 32
#define QUANT_R_Q8_0 1

struct block_q8_0
{
    float16_t d;
    int8_t qs[32];
};

struct block_q8_0_packed16
{
    float16_t d;
    int16_t qs[32/2];
};

#if defined(DATA_A_Q8_0)
#define QUANT_K QUANT_K_Q8_0
#define QUANT_R QUANT_R_Q8_0
#define QUANT_AUXF 1
#define A_TYPE block_q8_0
#define A_TYPE_PACKED16 block_q8_0_packed16
#define DATA_A_QUANT_LEGACY
#endif

#define QUANT_K_Q1_0 128
#define QUANT_R_Q1_0 1

struct block_q1_0
{
    float16_t d;
    uint8_t qs[QUANT_K_Q1_0 / 8];
};

#if defined(DATA_A_Q1_0)
#define QUANT_K QUANT_K_Q1_0
#define QUANT_R QUANT_R_Q1_0
#define QUANT_AUXF 1
#define A_TYPE block_q1_0
#endif

#define QUANT_K_Q2_0 64
#define QUANT_R_Q2_0 1

struct block_q2_0
{
    float16_t d;
    uint8_t qs[QUANT_K_Q2_0 / 4];
};

struct block_q2_0_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_Q2_0 / 8];
};

#if defined(DATA_A_Q2_0)
#define QUANT_K QUANT_K_Q2_0
#define QUANT_R QUANT_R_Q2_0
#define QUANT_AUXF 1
#define A_TYPE block_q2_0
#define A_TYPE_PACKED16 block_q2_0_packed16
#define DATA_A_QUANT_LEGACY
#endif

#define QUANT_K_Q8_1 32
#define QUANT_R_Q8_1 1

struct block_q8_1
{
    f16vec2 ds;
    int8_t qs[32];
};

struct block_q8_1_packed16
{
    f16vec2 ds;
    int16_t qs[16];
};

struct block_q8_1_packed32
{
    f16vec2 ds;
    int32_t qs[8];
};

// 4 blocks in one to allow 16-byte/128-bit alignment and loads
struct block_q8_1_x4
{
    f16vec2 ds[4];
    int32_t qs[32];
};

struct block_q8_1_x4_packed128
{
    f16vec2 ds[4];
    ivec4 qs[8];
};

// K-quants
#define QUANT_K_Q2_K 256

struct block_q2_K
{
    uint8_t scales[QUANT_K_Q2_K/16];
    uint8_t qs[QUANT_K_Q2_K/4];
    f16vec2 dm;
};

struct block_q2_K_packed16
{
    uint16_t scales[QUANT_K_Q2_K/16/2];
    uint16_t qs[QUANT_K_Q2_K/4/2];
    f16vec2 dm;
};

struct block_q2_K_packed32
{
    uint32_t scales[QUANT_K_Q2_K/16/4];
    uint32_t qs[QUANT_K_Q2_K/4/4];
    f16vec2 dm;
};

#if defined(DATA_A_Q2_K)
#define QUANT_K QUANT_K_Q2_K
#define QUANT_R 1
#define A_TYPE block_q2_K
#define A_TYPE_PACKED16 block_q2_K_packed16
#define A_TYPE_PACKED32 block_q2_K_packed32
#define SCALES_PER_32 2
#define DATA_A_QUANT_K
#endif

#define QUANT_K_TQ1_0 256

// TQ1_0: base-3 packed trits, 5 per byte in `qs` (48B) and 4 in `qh` (4B).
struct block_tq1_0
{
    uint8_t qs[(QUANT_K_TQ1_0 - 4 * QUANT_K_TQ1_0 / 64) / 5];
    uint8_t qh[QUANT_K_TQ1_0 / 64];
    float16_t d;
};

// Element e in [0,255] -> its packed byte (0..47 qs, 48..51 qh) and digit.
uint tq1_0_byte_of(uint e) {
    return e < 160u ? (e % 32u)
         : e < 240u ? 32u + ((e - 160u) % 16u)
         : 48u + ((e - 240u) % 4u);
}
uint tq1_0_digit_of(uint e) {
    return e < 160u ? (e / 32u)
         : e < 240u ? ((e - 160u) / 16u)
         : ((e - 240u) / 4u);
}
// The 8-bit truncation below is part of the format, not an optimisation:
// the C reference does `uint8_t q = qs[..] * pow3[n]`.
uint tq1_0_trit(uint qbyte, uint t) {
    const uint POW3_PACKED = (1u << 28) | (3u << 21) | (9u << 14) | (27u << 7) | 81u;
    return ((((qbyte * ((POW3_PACKED >> (7u * (4u - t))) & 0x7Fu)) & 255u) * 3u) >> 8);
}

#if defined(DATA_A_TQ1_0)
#define QUANT_K QUANT_K_TQ1_0
#define QUANT_R 1
#define A_TYPE block_tq1_0
#define DATA_A_QUANT_K
#endif

#define QUANT_K_TQ2_0 256

// ternary (BitNet): 2-bit codes, w = (q - 1) * d; qs layout matches q2_K's
// two 32-byte groups with four bit-levels per byte
struct block_tq2_0
{
    uint8_t qs[QUANT_K_TQ2_0/4];
    float16_t d;
};

struct block_tq2_0_packed16
{
    uint16_t qs[QUANT_K_TQ2_0/4/2];
    float16_t d;
};

#if defined(DATA_A_TQ2_0)
#define QUANT_K QUANT_K_TQ2_0
#define QUANT_R 1
#define A_TYPE block_tq2_0
#define A_TYPE_PACKED16 block_tq2_0_packed16
#define DATA_A_QUANT_K
#endif

#define QUANT_K_Q3_K 256

struct block_q3_K
{
    uint8_t hmask[QUANT_K_Q3_K/8];
    uint8_t qs[QUANT_K_Q3_K/4];
    uint8_t scales[12];
    float16_t d;
};

struct block_q3_K_packed16
{
    uint16_t hmask[QUANT_K_Q3_K/8/2];
    uint16_t qs[QUANT_K_Q3_K/4/2];
    uint16_t scales[12/2];
    float16_t d;
};

#if defined(DATA_A_Q3_K)
#define QUANT_K QUANT_K_Q3_K
#define QUANT_R 1
#define A_TYPE block_q3_K
#define A_TYPE_PACKED16 block_q3_K_packed16
#define DATA_A_QUANT_K
#endif

#define QUANT_K_Q4_K 256

struct block_q4_K
{
    f16vec2 dm;
    uint8_t scales[3*QUANT_K_Q4_K/64];
    uint8_t qs[QUANT_K_Q4_K/2];
};

struct block_q4_K_packed16
{
    f16vec2 dm;
    uint16_t scales[3*QUANT_K_Q4_K/64/2];
    uint16_t qs[QUANT_K_Q4_K/2/2];
};

struct block_q4_K_packed32
{
    f16vec2 dm;
    uint32_t scales[3*QUANT_K_Q4_K/64/4];
    uint32_t qs[QUANT_K_Q4_K/2/4];
};

struct block_q4_K_packed128
{
    uvec4 q4k[9];
};

#if defined(DATA_A_Q4_K)
#define QUANT_K QUANT_K_Q4_K
#define QUANT_R 1
#define A_TYPE block_q4_K
#define A_TYPE_PACKED16 block_q4_K_packed16
#define A_TYPE_PACKED32 block_q4_K_packed32
#define DATA_A_QUANT_K
#endif

#define QUANT_K_Q5_K 256

struct block_q5_K
{
    f16vec2 dm;
    uint8_t scales[12];
    uint8_t qh[QUANT_K_Q5_K/8];
    uint8_t qs[QUANT_K_Q5_K/2];
};

struct block_q5_K_packed16
{
    f16vec2 dm;
    uint16_t scales[12/2];
    uint16_t qh[QUANT_K_Q5_K/8/2];
    uint16_t qs[QUANT_K_Q5_K/2/2];
};

struct block_q5_K_packed32
{
    f16vec2 dm;
    uint32_t scales[12/4];
    uint32_t qh[QUANT_K_Q5_K/8/4];
    uint32_t qs[QUANT_K_Q5_K/2/4];
};

struct block_q5_K_packed128
{
    uvec4 q5k[11];
};

#if defined(DATA_A_Q5_K)
#define QUANT_K QUANT_K_Q5_K
#define QUANT_R 1
#define A_TYPE block_q5_K
#define A_TYPE_PACKED16 block_q5_K_packed16
#define A_TYPE_PACKED32 block_q5_K_packed32
#define DATA_A_QUANT_K
#endif

#define QUANT_K_Q6_K 256

struct block_q6_K
{
    uint8_t ql[QUANT_K_Q6_K/2];
    uint8_t qh[QUANT_K_Q6_K/4];
    int8_t scales[QUANT_K_Q6_K/16];
    float16_t d;
};

struct block_q6_K_packed16
{
    uint16_t ql[QUANT_K_Q6_K/2/2];
    uint16_t qh[QUANT_K_Q6_K/4/2];
    int16_t scales[QUANT_K_Q6_K/16/2];
    float16_t d;
};

#if defined(DATA_A_Q6_K)
#define QUANT_K QUANT_K_Q6_K
#define QUANT_R 1
#define A_TYPE block_q6_K
#define A_TYPE_PACKED16 block_q6_K_packed16
#define DATA_A_QUANT_K
#endif

// IQuants

#define QUANT_K_IQ1_S 256
#define QUANT_R_IQ1_S 1

struct block_iq1_s {
    float16_t d;
    uint8_t  qs[QUANT_K_IQ1_S/8];
    uint16_t qh[QUANT_K_IQ1_S/32];
};

struct block_iq1_s_packed16 {
    float16_t d;
    uint16_t qs[QUANT_K_IQ1_S/8/2];
    uint16_t qh[QUANT_K_IQ1_S/32];
};

#define QUANT_K_IQ1_M 256
#define QUANT_R_IQ1_M 1

struct block_iq1_m {
    uint8_t  qs[QUANT_K_IQ1_M/8];
    uint8_t  qh[QUANT_K_IQ1_M/16];
    uint16_t scales[QUANT_K_IQ1_M/64];
};

struct block_iq1_m_packed16 {
    uint16_t qs[QUANT_K_IQ1_M/8/2];
    uint16_t qh[QUANT_K_IQ1_M/16/2];
    uint16_t scales[QUANT_K_IQ1_M/64];
};

struct block_iq1_m_packed32 {
    uint32_t qs[QUANT_K_IQ1_M/8/4];
    uint32_t qh[QUANT_K_IQ1_M/16/4];
    uint32_t scales[QUANT_K_IQ1_M/64/2];
};

struct block_iq1_m_packed64 {
    uint64_t  qs[QUANT_K_IQ1_M/8/8];
    uint64_t  qh[QUANT_K_IQ1_M/16/8];
    uint64_t scales;
};

#if defined(DATA_A_IQ1_S)
#define QUANT_K QUANT_K_IQ1_S
#define QUANT_R QUANT_R_IQ1_S
#define A_TYPE block_iq1_s
#define A_TYPE_PACKED16 block_iq1_s_packed16
#endif

#if defined(DATA_A_IQ1_M)
#define QUANT_K QUANT_K_IQ1_M
#define QUANT_R QUANT_R_IQ1_M
#define A_TYPE block_iq1_m
#define A_TYPE_PACKED16 block_iq1_m_packed16
#define A_TYPE_PACKED32 block_iq1_m_packed32
#endif

// IQ1/IQ2/IQ3 grid tables are read from a storage buffer and copied to shared memory by init_iq_shmem.
// The binding and offsets must match IQ_GRID_BINDING and vk_iq_grids in ggml-vulkan.
#define IQ_GRID_BINDING 12
// Offsets are in bytes; divide by the element size when indexing.
#define IQ1S_GRID_OFFSET     0
#define IQ1S_GRID_GPU_OFFSET 4096
#define IQ2XXS_GRID_OFFSET   12288
#define IQ2XS_GRID_OFFSET    14336
#define IQ2S_GRID_OFFSET     18432
#define IQ3XXS_GRID_OFFSET   26624
#define IQ3S_GRID_OFFSET     27648

#if defined(DATA_A_IQ1_S) || defined(DATA_A_IQ1_M)
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

layout (binding = IQ_GRID_BINDING) readonly buffer IQGrid { uint32_t iq_grid[]; };

#define IQ1S_GRID_SIZE 2048
// The buffer packs two 16-bit grid entries per uint32, so it needs no 16-bit storage buffer support
#define IQ1S_GRID_PACKED_LEN (IQ1S_GRID_SIZE / 2)

shared uint16_t iq1s_grid[IQ1S_GRID_SIZE];
#if defined(NEEDS_IQ1S_GRID_GPU)
shared uint32_t iq1s_grid_gpu[IQ1S_GRID_SIZE];
#endif

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < IQ1S_GRID_PACKED_LEN; i += wgsize.x) {
        uint idx = i + gl_LocalInvocationIndex.x;
        if (IQ1S_GRID_PACKED_LEN % wgsize.x == 0 || idx < IQ1S_GRID_PACKED_LEN) {
            u16vec2 g = unpack16(iq_grid[IQ1S_GRID_OFFSET / 4 + idx]);  // bytes to uint32 index
            iq1s_grid[2*idx+0] = g.x;
            iq1s_grid[2*idx+1] = g.y;
        }
    }
#if defined(NEEDS_IQ1S_GRID_GPU)
    [[unroll]] for (uint i = 0; i < iq1s_grid_gpu.length(); i += wgsize.x) {
        uint idx = i + gl_LocalInvocationIndex.x;
        if (iq1s_grid_gpu.length() % wgsize.x == 0 || idx < iq1s_grid_gpu.length()) {
            iq1s_grid_gpu[idx] = iq_grid[IQ1S_GRID_GPU_OFFSET / 4 + idx];  // bytes to uint32 index
        }
    }
#endif
    barrier();
}
#endif

#if defined(DATA_A_IQ2_XXS) || defined(DATA_A_IQ2_XS) || defined(DATA_A_IQ2_S)
layout (binding = IQ_GRID_BINDING) readonly buffer IQGrid { uvec2 iq_grid[]; };

#if defined(DATA_A_IQ2_S)
shared uvec2 iq2s_grid[1024];
#elif defined(DATA_A_IQ2_XS)
shared uvec2 iq2xs_grid[512];
#else
shared uvec2 iq2xxs_grid[256];
#endif
#endif

#define QUANT_K_IQ2_XXS 256
#define QUANT_R_IQ2_XXS 1

struct block_iq2_xxs
{
    float16_t d;
    uint8_t qs[QUANT_K_IQ2_XXS/4];
};

struct block_iq2_xxs_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ2_XXS/8];
};

#if defined(DATA_A_IQ2_XXS)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < iq2xxs_grid.length(); i += wgsize.x) {
        if (iq2xxs_grid.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2xxs_grid.length()) {
            iq2xxs_grid[i + gl_LocalInvocationIndex.x] = iq_grid[IQ2XXS_GRID_OFFSET / 8 + i + gl_LocalInvocationIndex.x];  // bytes to uvec2 index
        }
    }
    barrier();
}

#define QUANT_K QUANT_K_IQ2_XXS
#define QUANT_R QUANT_R_IQ2_XXS
#define A_TYPE block_iq2_xxs
#define A_TYPE_PACKED16 block_iq2_xxs_packed16
#endif

#define QUANT_K_IQ2_XS 256
#define QUANT_R_IQ2_XS 1

struct block_iq2_xs
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ2_XS/8];
    uint8_t scales[QUANT_K_IQ2_XS/32];
};

struct block_iq2_xs_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ2_XS/8];
    uint16_t scales[QUANT_K_IQ2_XS/64];
};

#if defined(DATA_A_IQ2_XS)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < iq2xs_grid.length(); i += wgsize.x) {
        if (iq2xs_grid.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2xs_grid.length()) {
            iq2xs_grid[i + gl_LocalInvocationIndex.x] = iq_grid[IQ2XS_GRID_OFFSET / 8 + i + gl_LocalInvocationIndex.x];  // bytes to uvec2 index
        }
    }
    barrier();
}

#define QUANT_K QUANT_K_IQ2_XS
#define QUANT_R QUANT_R_IQ2_XS
#define A_TYPE block_iq2_xs
#define A_TYPE_PACKED16 block_iq2_xs_packed16
#endif

#define QUANT_K_IQ2_S 256
#define QUANT_R_IQ2_S 1

struct block_iq2_s
{
    float16_t d;
    uint8_t qs[QUANT_K_IQ2_S/4];
    uint8_t qh[QUANT_K_IQ2_S/32];
    uint8_t scales[QUANT_K_IQ2_S/32];
};

struct block_iq2_s_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ2_S/8];
    uint16_t qh[QUANT_K_IQ2_S/64];
    uint16_t scales[QUANT_K_IQ2_S/64];
};

#if defined(DATA_A_IQ2_S)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < iq2s_grid.length(); i += wgsize.x) {
        if (iq2s_grid.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2s_grid.length()) {
            iq2s_grid[i + gl_LocalInvocationIndex.x] = iq_grid[IQ2S_GRID_OFFSET / 8 + i + gl_LocalInvocationIndex.x];  // bytes to uvec2 index
        }
    }
    barrier();
}

#define QUANT_K QUANT_K_IQ2_S
#define QUANT_R QUANT_R_IQ2_S
#define A_TYPE block_iq2_s
#define A_TYPE_PACKED16 block_iq2_s_packed16
#endif

#if defined(DATA_A_IQ3_XXS) || defined(DATA_A_IQ3_S)
layout (binding = IQ_GRID_BINDING) readonly buffer IQGrid { uint32_t iq_grid[]; };

#if defined(DATA_A_IQ3_S)
shared uint32_t iq3s_grid[512];
#else
shared uint32_t iq3xxs_grid[256];
#endif
#endif

#define QUANT_K_IQ3_XXS 256
#define QUANT_R_IQ3_XXS 1

struct block_iq3_xxs
{
    float16_t d;
    uint8_t qs[QUANT_K_IQ3_XXS/4 + QUANT_K_IQ3_XXS/8];
};

struct block_iq3_xxs_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ3_XXS/8 + QUANT_K_IQ3_XXS/16];
};

#if defined(DATA_A_IQ3_XXS)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < iq3xxs_grid.length(); i += wgsize.x) {
        if (iq3xxs_grid.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq3xxs_grid.length()) {
            iq3xxs_grid[i + gl_LocalInvocationIndex.x] = iq_grid[IQ3XXS_GRID_OFFSET / 4 + i + gl_LocalInvocationIndex.x];  // bytes to uint32 index
        }
    }
    barrier();
}

#define QUANT_K QUANT_K_IQ3_XXS
#define QUANT_R QUANT_R_IQ3_XXS
#define A_TYPE block_iq3_xxs
#define A_TYPE_PACKED16 block_iq3_xxs_packed16
#endif

#define QUANT_K_IQ3_S 256
#define QUANT_R_IQ3_S 1

struct block_iq3_s
{
    float16_t d;
    uint8_t qs[QUANT_K_IQ3_S/4];
    uint8_t qh[QUANT_K_IQ3_S/32];
    uint8_t signs[QUANT_K_IQ3_S/8];
    uint8_t scales[QUANT_K_IQ3_S/64];
};

struct block_iq3_s_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ3_S/4/2];
    uint16_t qh[QUANT_K_IQ3_S/32/2];
    uint16_t signs[QUANT_K_IQ3_S/8/2];
    uint16_t scales[QUANT_K_IQ3_S/64/2];
};

#if defined(DATA_A_IQ3_S)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < iq3s_grid.length(); i += wgsize.x) {
        if (iq3s_grid.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq3s_grid.length()) {
            iq3s_grid[i + gl_LocalInvocationIndex.x] = iq_grid[IQ3S_GRID_OFFSET / 4 + i + gl_LocalInvocationIndex.x];  // bytes to uint32 index
        }
    }
    barrier();
}

#define QUANT_K QUANT_K_IQ3_S
#define QUANT_R QUANT_R_IQ3_S
#define A_TYPE block_iq3_s
#define A_TYPE_PACKED16 block_iq3_s_packed16
#endif

#define QUANT_K_IQ4_XS 256
#define QUANT_R_IQ4_XS 1

struct block_iq4_xs
{
    float16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QUANT_K_IQ4_XS/64];
    uint8_t qs[QUANT_K_IQ4_XS/2];
};

struct block_iq4_xs_packed16
{
    float16_t d;
    uint16_t scales_h;
    uint16_t scales_l[QUANT_K_IQ4_XS/128];
    uint16_t qs[QUANT_K_IQ4_XS/4];
};

struct block_iq4_xs_packed32
{
    float16_t d;
    uint16_t scales_h;
    uint32_t scales_l;
    uint32_t qs[QUANT_K_IQ4_XS/8];
};

#if defined(DATA_A_IQ4_XS)
#define QUANT_K QUANT_K_IQ4_XS
#define QUANT_R QUANT_R_IQ4_XS
#define A_TYPE block_iq4_xs
#define A_TYPE_PACKED16 block_iq4_xs_packed16
#define A_TYPE_PACKED32 block_iq4_xs_packed32
#endif

#define QUANT_K_IQ4_NL 32
#define QUANT_R_IQ4_NL 2

struct block_iq4_nl
{
    float16_t d;
    uint8_t qs[QUANT_K_IQ4_NL/2];
};

struct block_iq4_nl_packed16
{
    float16_t d;
    uint16_t qs[QUANT_K_IQ4_NL/2/2];
};

#if defined(DATA_A_IQ4_NL)
#define QUANT_K QUANT_K_IQ4_NL
#define QUANT_R QUANT_R_IQ4_NL
#define QUANT_AUXF 1
#define A_TYPE block_iq4_nl
#define A_TYPE_PACKED16 block_iq4_nl_packed16
#endif

#define QUANT_K_MXFP4 32
#define QUANT_R_MXFP4 2

struct block_mxfp4
{
    uint8_t e;
    uint8_t qs[QUANT_K_MXFP4/2];
};

#if defined(DATA_A_MXFP4)
#define QUANT_K QUANT_K_MXFP4
#define QUANT_R QUANT_R_MXFP4
#define QUANT_AUXF 1
#define A_TYPE block_mxfp4
#endif

#define QUANT_K_NVFP4 64
#define QUANT_R_NVFP4 1

struct block_nvfp4
{
    uint8_t d[QUANT_K_NVFP4 / 16];
    uint8_t qs[QUANT_K_NVFP4 / 2];
};

struct block_nvfp4_packed16
{
    uint16_t d[QUANT_K_NVFP4 / 16 / 2];
    uint16_t qs[QUANT_K_NVFP4 / 2 / 2];
};

struct block_nvfp4_packed32
{
    uint32_t d[QUANT_K_NVFP4 / 16 / 4];
    uint32_t qs[QUANT_K_NVFP4 / 2 / 4];
};

#if defined(DATA_A_NVFP4)
#define QUANT_K QUANT_K_NVFP4
#define QUANT_R QUANT_R_NVFP4
#define QUANT_AUXF 1
#define A_TYPE block_nvfp4
#define A_TYPE_PACKED16 block_nvfp4_packed16
#define A_TYPE_PACKED32 block_nvfp4_packed32
#endif

#if defined(DATA_A_IQ4_NL) || defined(DATA_A_IQ4_XS)
const int8_t kvalues_iq4nl_const[16] = {
    int8_t(-127), int8_t(-104), int8_t(-83), int8_t(-65), int8_t(-49), int8_t(-35), int8_t(-22), int8_t(-10),
    int8_t(1), int8_t(13), int8_t(25), int8_t(38), int8_t(53), int8_t(69), int8_t(89), int8_t(113)
};

#ifdef KVALUES_IQ4NL_I8
shared int8_t kvalues_iq4nl[16];
#else
shared FLOAT_TYPE kvalues_iq4nl[16];
#endif

#if defined(DATA_A_IQ4_NL) || defined(DATA_A_IQ4_XS)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    for (uint i = gl_LocalInvocationIndex.x; i < kvalues_iq4nl.length(); i += wgsize.x) {
#ifdef KVALUES_IQ4NL_I8
        kvalues_iq4nl[i] = kvalues_iq4nl_const[i];
#else
        kvalues_iq4nl[i] = FLOAT_TYPE(kvalues_iq4nl_const[i]);
#endif
    }
    barrier();
}

#ifdef KVALUES_IQ4NL_I8
i32vec2 iq4nl_to_i8x8(uint32_t vui) {
    const u8vec4 i0 = unpack8( vui       & 0x0F0F0F0F);
    const u8vec4 i1 = unpack8((vui >> 4) & 0x0F0F0F0F);

    return i32vec2(
        pack32(i8vec4(kvalues_iq4nl[i0.x], kvalues_iq4nl[i0.y], kvalues_iq4nl[i0.z], kvalues_iq4nl[i0.w])),
        pack32(i8vec4(kvalues_iq4nl[i1.x], kvalues_iq4nl[i1.y], kvalues_iq4nl[i1.z], kvalues_iq4nl[i1.w])));
}
#endif
#endif
#endif

#if defined(DATA_A_MXFP4) || defined(DATA_A_NVFP4)
#if !defined(USE_OCP_FP4)
const int8_t kvalues_mxfp4_const[16] = {
    int8_t(0), int8_t(1), int8_t(2), int8_t(3), int8_t(4), int8_t(6), int8_t(8), int8_t(12),
    int8_t(0), int8_t(-1), int8_t(-2), int8_t(-3), int8_t(-4), int8_t(-6), int8_t(-8), int8_t(-12),
};

shared int8_t kvalues_mxfp4[16];
#endif

#if defined(DATA_A_NVFP4) && !defined(USE_OCP_FP4)
// Cover all scale bytes accepted by ggml_validate_row_data.
// ue4m3_to_fp32_build() masks the exponent and mantissa, but checks the full byte for 0 and 0x7F.
shared float ue4m3_fp32_lut[256];

float ue4m3_to_fp32_build(uint u) {
    if (u == 0u || u == 127u) {
        return 0.0;
    }
    const uint exp = (u >> 3) & 15u;
    const uint man = u & 7u;
    if (exp == 0u) {
        return float(man) * (1.0 / 512.0);
    }
    const uint bits = (exp + 120u) << 23 | (man << 20);
    return uintBitsToFloat(bits);
}
#endif

#if (defined(DATA_A_MXFP4) || defined(DATA_A_NVFP4)) && !defined(USE_OCP_FP4)
#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    for (uint i = gl_LocalInvocationIndex.x; i < kvalues_mxfp4.length(); i += wgsize.x) {
        kvalues_mxfp4[i] = kvalues_mxfp4_const[i];
    }
#if defined(DATA_A_NVFP4)
    for (uint i = gl_LocalInvocationIndex.x; i < 256u; i += wgsize.x) {
        ue4m3_fp32_lut[i] = ue4m3_to_fp32_build(i);
    }
#endif
    barrier();
}
#endif
#endif

// returns the bfloat value in the low 16b.
// See ggml_compute_fp32_to_bf16
uint32_t fp32_to_bf16(float f)
{
    uint32_t u = floatBitsToUint(f);
    u = (u + (0x7fff + ((u >> 16) & 1))) >> 16;
    return u;
}

float bf16_to_fp32(uint32_t u)
{
    return uintBitsToFloat(u << 16);
}

vec4 bf16_to_fp32(uvec4 u)
{
    return vec4(bf16_to_fp32(u.x), bf16_to_fp32(u.y), bf16_to_fp32(u.z), bf16_to_fp32(u.w));
}

#if defined(DATA_A_BF16)
#define A_TO_FLOAT(x) bf16_to_fp32(uint32_t(x))
#else
#define A_TO_FLOAT(x) float(x)
#endif
#if defined(DATA_B_BF16)
#define B_TO_FLOAT(x) bf16_to_fp32(uint32_t(x))
#else
#define B_TO_FLOAT(x) float(x)
#endif
#if defined(DATA_C_BF16)
#define C_TO_FLOAT(x) bf16_to_fp32(uint32_t(x))
#else
#define C_TO_FLOAT(x) float(x)
#endif
#if defined(DATA_D_BF16)
#define FLOAT_TO_D(x) D_TYPE(fp32_to_bf16(x))
#else
#define FLOAT_TO_D(x) D_TYPE(x)
#endif

float e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = x;
        bits = bits << 23;
    }

    return uintBitsToFloat(bits);
}

#if defined(DATA_A_NVFP4)
#if defined(USE_OCP_FP4)
floate4m3_t ue4m3_from_bits(uint8_t x) {
    if (x == uint8_t(0x7F)) {
        return floate4m3_t(0.0);
    }
    return uintBitsToFloate4m3EXT(x);
}
#endif

float ue4m3_to_fp32(uint8_t x) {
#if defined(USE_OCP_FP4)
    return float(ue4m3_from_bits(x));
#else
    return ue4m3_fp32_lut[uint(x)];
#endif
}
#endif

#if BDA

#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#define BDA_STORAGE_T uint64_t
#define BDA_OFFSET_T uint64_t

#else

#define BDA_STORAGE_T uvec2
#define BDA_OFFSET_T uint

#endif

#endif // !defined(GGML_TYPES_COMP)
