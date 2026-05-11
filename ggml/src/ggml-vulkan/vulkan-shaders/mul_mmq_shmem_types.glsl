#if defined(FA_MMQ_MIXED)
// Mixed-K flash attention MMQ: superset cache that fits Q4_0/Q4_1/Q5_0/Q5_1/Q8_0.
// Q4_*/Q5_* only use qs[0..3] and (for Q5_*) qh. Q8_0 uses qs[0..7]. Single-scale
// types (Q4_0/Q5_0/Q8_0) leave dm.y unused.
struct block_a_cache {
    int32_t qs[8];
    uint32_t qh;
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q4_0)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_Q4_1)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[16/4];
    FLOAT_TYPEV2 dm;
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
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q8_0)
#define QUANT_R_MMQ 1
// AMD likes 4, Intel likes 1 and Nvidia likes 2
// #define BK_STEP 1
struct block_a_cache {
    int32_t qs[32/4];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_IQ4_NL)
#define QUANT_R_MMQ 2
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPE dm;
};
#elif defined(DATA_A_MXFP4)
#define QUANT_R_MMQ 2
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPE d;
};
#elif defined(DATA_A_Q2_K)
#define QUANT_R_MMQ 4
struct block_a_cache {
    uint32_t qs[2];
    u8vec2 scales;
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q3_K)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[4];
    FLOAT_TYPEV2 d_scales;
};
#elif defined(DATA_A_Q4_K)
#define QUANT_R_MMQ 2
struct block_a_cache {
    uint32_t qs[4];
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q5_K)
#define QUANT_R_MMQ 1
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPEV2 dm;
};
#elif defined(DATA_A_Q6_K)
#define QUANT_R_MMQ 1
struct block_a_cache {
    int32_t qs[8];
    FLOAT_TYPEV2 d_scales;
};
#endif

struct block_b_cache
{
    int32_t qs[8];
    FLOAT_TYPEV2 ds;
};
