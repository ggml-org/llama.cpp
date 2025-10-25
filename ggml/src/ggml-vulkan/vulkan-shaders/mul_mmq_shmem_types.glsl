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
#define QUANT_R_MMQ 4
struct block_a_cache
{
    uint32_t qs[2];
    u8vec2 scales;
    FLOAT_TYPE_VEC2 dm;
};
#endif

struct block_b_cache
{
    int32_t qs[8];
    FLOAT_TYPE_VEC2 ds;
};
