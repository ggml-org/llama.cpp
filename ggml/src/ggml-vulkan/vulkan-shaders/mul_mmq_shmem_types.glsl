#if defined(DATA_A_Q4_0)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_Q4_1)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    FLOAT_TYPE_VEC2 dm;
};
#elif defined(DATA_A_Q5_0)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    uint32_t qh;
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_Q5_1)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    uint32_t qh;
    FLOAT_TYPE_VEC2 dm;
};
#elif defined(DATA_A_Q8_0)
#define QUANT_R_MMQ 1
struct block_a_cache {
    int32_t qs[32/4];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_Q2_K)
#define QUANT_R_MMQ 1
struct block_a_cache
{
    uint32_t qs[8];
    u8vec4 scales[2];
    FLOAT_TYPE_VEC2 dm;
};
#endif

#if defined(DATA_A_QUANT_LEGACY)
#define QUANT_BLOCK_FACTOR 1

struct block_b_cache
{
    int32_t qs[8];
    FLOAT_TYPE_VEC2 ds;
};
#elif defined(DATA_A_QUANT_K)
#define QUANT_BLOCK_FACTOR 4

struct block_b_cache
{
    int32_t qs[32];
    FLOAT_TYPE_VEC2 ds[4];
};
#else
#error unimplemented
#endif
