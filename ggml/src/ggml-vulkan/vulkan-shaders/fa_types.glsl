#if !defined(GGML_FA_TYPES_COMP)
#define GGML_FA_TYPES_COMP

// FaTypeK / FaTypeV spec constant values. These mirror enum ggml_type so the
// host can pass the type directly. Keep in sync with ggml.h.
#define FA_TYPE_F32   0u
#define FA_TYPE_F16   1u
#define FA_TYPE_Q4_0  2u
#define FA_TYPE_Q4_1  3u
#define FA_TYPE_Q5_0  6u
#define FA_TYPE_Q5_1  7u
#define FA_TYPE_Q8_0  8u
#define FA_TYPE_IQ4_NL 20u
#define FA_TYPE_BF16 30u

// Number of matrix elements per buffer block, derived from the K/V type spec
// constant. F32 is treated as a vec4 "block" of 4 floats. F16 uses block size 1
// and bypasses the dequant path entirely. Quants follow their ggml block sizes.
uint fa_block_elems(uint ty) {
    switch (ty) {
        case FA_TYPE_F32:  return 4u;
        case FA_TYPE_F16:  return 1u;
        case FA_TYPE_Q4_0: return uint(QUANT_K_Q4_0);
        case FA_TYPE_Q4_1: return uint(QUANT_K_Q4_1);
        case FA_TYPE_Q5_0: return uint(QUANT_K_Q5_0);
        case FA_TYPE_Q5_1: return uint(QUANT_K_Q5_1);
        case FA_TYPE_Q8_0: return uint(QUANT_K_Q8_0);
        case FA_TYPE_IQ4_NL: return uint(QUANT_K_IQ4_NL);
        case FA_TYPE_BF16: return 1u;
        default:           return 1u;
    }
}

// QUANT_R_MMQ for FA-eligible K types. Q4_*/Q5_* store two nibbles per byte
// (R==2); Q8_0 stores one byte per element (R==1). Used to derive the number
// of int32s per 32-element block on the MMQ K path: ints_per_block == 8 / R.
uint fa_quant_r_mmq(uint ty) {
    switch (ty) {
        case FA_TYPE_Q4_0: return uint(QUANT_R_Q4_0);
        case FA_TYPE_Q4_1: return uint(QUANT_R_Q4_1);
        case FA_TYPE_Q5_0: return uint(QUANT_R_Q5_0);
        case FA_TYPE_Q5_1: return uint(QUANT_R_Q5_1);
        case FA_TYPE_Q8_0: return uint(QUANT_R_Q8_0);
        default:           return 1u;
    }
}

bool fa_type_needs_shmem(uint ty) {
    switch (ty) {
        case FA_TYPE_IQ4_NL: return true;
        default:             return false;
    }
}

#endif // !defined(GGML_FA_TYPES_COMP)
