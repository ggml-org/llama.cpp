// MMQ K-side helpers, asymmetric form. Each function dispatches on FaTypeK and
// reads from the matching aliased K binding declared in flash_attn_dequant.glsl.
// Spec-constant specialization folds the unused paths.

int32_t get_k_qs(uint ib, uint iqs, uint a_offset) {
    switch (FaTypeK) {
        case FA_TYPE_Q4_0: {
            uint vui = pack32(u16vec2(k_packed_q4_0.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0],
                                      k_packed_q4_0.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]));
            uint shift = (iqs & 0x10) >> 2;
            vui >>= shift;
            return int32_t(vui & 0x0F0F0F0F);
        }
        case FA_TYPE_Q4_1: { // uses packed32 alias
            uint vui = k_packed_q4_1_p32.data[a_offset + ib].qs[(iqs & 0xF) / 4];
            uint shift = (iqs & 0x10) >> 2;
            vui >>= shift;
            return int32_t(vui & 0x0F0F0F0F);
        }
        case FA_TYPE_Q5_0: {
            uint vui = pack32(u16vec2(k_packed_q5_0.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0],
                                      k_packed_q5_0.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]));
            uint qh = pack32(u16vec2(k_packed_q5_0.data[a_offset + ib].qh[0],
                                     k_packed_q5_0.data[a_offset + ib].qh[1]));
            uint shift = (iqs & 0x10) >> 2;
            vui >>= shift;
            uint qh_bits = (qh >> iqs) & 0xF;
            return int32_t(vui & 0x0F0F0F0F) | int32_t((qh_bits * 0x02040810u) & 0x10101010u);
        }
        case FA_TYPE_Q5_1: { // qs via packed32, qh via packed16
            uint vui = k_packed_q5_1_p32.data[a_offset + ib].qs[(iqs & 0xF) / 4];
            uint qh  = k_packed_q5_1.data[a_offset + ib].qh;
            uint shift = (iqs & 0x10) >> 2;
            vui >>= shift;
            uint qh_bits = (qh >> iqs) & 0xF;
            return int32_t(vui & 0x0F0F0F0F) | int32_t((qh_bits * 0x02040810u) & 0x10101010u);
        }
        case FA_TYPE_Q8_0: {
            return pack32(i16vec2(k_packed_q8_0.data[a_offset + ib].qs[iqs / 2],
                                  k_packed_q8_0.data[a_offset + ib].qs[iqs / 2 + 1]));
        }
        default: return 0;
    }
}

// Per-block scale/min, packed as (d, m). Single-scale types (Q4_0, Q5_0, Q8_0)
// return (d, 0) so call sites always see the same shape.
FLOAT_TYPEV2 get_k_scale(uint ib, uint a_offset) {
    switch (FaTypeK) {
        case FA_TYPE_Q4_0: return FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q4_0.data[a_offset + ib].d), 0.0);
        case FA_TYPE_Q4_1: return FLOAT_TYPEV2(k_packed_q4_1_p32.data[a_offset + ib].dm);
        case FA_TYPE_Q5_0: return FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q5_0.data[a_offset + ib].d), 0.0);
        case FA_TYPE_Q5_1: return FLOAT_TYPEV2(k_packed_q5_1_p32.data[a_offset + ib].dm);
        case FA_TYPE_Q8_0: return FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q8_0.data[a_offset + ib].d), 0.0);
        default: return FLOAT_TYPEV2(0);
    }
}

void k_block_to_shmem(const uint buf_ib, const uint global_ib, const uint iqs, const uint a_offset) {
    // kblocksh[].qs is int32_t for the unified MMQ struct; uint sources need
    // explicit casts. The bit pattern is what we care about here -- the actual
    // signed/unsigned interpretation happens downstream in the dot product.
    switch (FaTypeK) {
        case FA_TYPE_Q4_0: {
            kblocksh[buf_ib].qs[iqs] = int32_t(pack32(u16vec2(k_packed_q4_0.data[a_offset + global_ib].qs[iqs * 2],
                                                              k_packed_q4_0.data[a_offset + global_ib].qs[iqs * 2 + 1])));
            break;
        }
        case FA_TYPE_Q4_1: {
            kblocksh[buf_ib].qs[iqs] = int32_t(k_packed_q4_1_p32.data[a_offset + global_ib].qs[iqs]);
            break;
        }
        case FA_TYPE_Q5_0: {
            kblocksh[buf_ib].qs[iqs] = int32_t(pack32(u16vec2(k_packed_q5_0.data[a_offset + global_ib].qs[iqs * 2],
                                                              k_packed_q5_0.data[a_offset + global_ib].qs[iqs * 2 + 1])));
            if (iqs == 0) {
                kblocksh[buf_ib].qh = pack32(u16vec2(k_packed_q5_0.data[a_offset + global_ib].qh[0],
                                                     k_packed_q5_0.data[a_offset + global_ib].qh[1]));
            }
            break;
        }
        case FA_TYPE_Q5_1: {
            kblocksh[buf_ib].qs[iqs] = int32_t(k_packed_q5_1_p32.data[a_offset + global_ib].qs[iqs]);
            if (iqs == 0) {
                kblocksh[buf_ib].qh = k_packed_q5_1.data[a_offset + global_ib].qh;
            }
            break;
        }
        case FA_TYPE_Q8_0: {
            kblocksh[buf_ib].qs[iqs] = pack32(i16vec2(k_packed_q8_0.data[a_offset + global_ib].qs[iqs * 2],
                                                      k_packed_q8_0.data[a_offset + global_ib].qs[iqs * 2 + 1]));
            break;
        }
    }

    if (iqs == 0) {
        // Q4_0/Q5_0/Q8_0 store dm.x = d; Q4_1/Q5_1 store dm = (d, m) pair.
        switch (FaTypeK) {
            case FA_TYPE_Q4_0: kblocksh[buf_ib].dm = FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q4_0.data[a_offset + global_ib].d), 0.0); break;
            case FA_TYPE_Q4_1: kblocksh[buf_ib].dm = FLOAT_TYPEV2(k_packed_q4_1_p32.data[a_offset + global_ib].dm); break;
            case FA_TYPE_Q5_0: kblocksh[buf_ib].dm = FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q5_0.data[a_offset + global_ib].d), 0.0); break;
            case FA_TYPE_Q5_1: kblocksh[buf_ib].dm = FLOAT_TYPEV2(k_packed_q5_1_p32.data[a_offset + global_ib].dm); break;
            case FA_TYPE_Q8_0: kblocksh[buf_ib].dm = FLOAT_TYPEV2(FLOAT_TYPE(k_packed_q8_0.data[a_offset + global_ib].d), 0.0); break;
        }
    }
}

// d_per_step==8 hot path: read one full 32-element block worth of nibble-packed
// int32 quants. Equivalent to 8 calls to get_k_qs(ib, d*4, a_offset) but reads
// qh (Q5_*) and runs pack32 (Q4_0/Q5_0) once per block instead of per nibble
// quad. iqs is always 0 in this path (hsk4 % 8 == 0 implies block-aligned).
// Q8_0 takes the generic get_k_qs path because its qs layout (i8 pairs) doesn't
// share this nibble shape.
//
// Returned via a struct so the caller's k_quants array (sized from spec
// constants) doesn't need to match a fixed[8] out-parameter type.
struct fa_k_qs_block8 {
    int32_t qs[8];
};

fa_k_qs_block8 get_k_qs_block8(uint ib, uint a_offset) {
    fa_k_qs_block8 r;
    uint qh = 0;
    if (FaTypeK == FA_TYPE_Q5_0) {
        qh = pack32(u16vec2(k_packed_q5_0.data[a_offset + ib].qh[0],
                            k_packed_q5_0.data[a_offset + ib].qh[1]));
    } else if (FaTypeK == FA_TYPE_Q5_1) {
        qh = k_packed_q5_1.data[a_offset + ib].qh;
    }
    const bool has_qh = (FaTypeK == FA_TYPE_Q5_0) || (FaTypeK == FA_TYPE_Q5_1);
    [[unroll]] for (uint32_t d = 0; d < 4; d++) {
        uint vui = 0;
        switch (FaTypeK) {
            case FA_TYPE_Q4_0: { // packed16
                vui = pack32(u16vec2(k_packed_q4_0.data[a_offset + ib].qs[d * 2 + 0],
                                     k_packed_q4_0.data[a_offset + ib].qs[d * 2 + 1]));
                break;
            }
            case FA_TYPE_Q4_1: { // packed32 alias
                vui = k_packed_q4_1_p32.data[a_offset + ib].qs[d];
                break;
            }
            case FA_TYPE_Q5_0: { // packed16
                vui = pack32(u16vec2(k_packed_q5_0.data[a_offset + ib].qs[d * 2 + 0],
                                     k_packed_q5_0.data[a_offset + ib].qs[d * 2 + 1]));
                break;
            }
            case FA_TYPE_Q5_1: { // packed32 alias
                vui = k_packed_q5_1_p32.data[a_offset + ib].qs[d];
                break;
            }
        }
        r.qs[d    ] = int32_t( vui       & 0x0F0F0F0F);
        r.qs[d + 4] = int32_t((vui >> 4) & 0x0F0F0F0F);
        if (has_qh) {
            uint qh_lo = (qh >> (d * 4))      & 0xFu;
            uint qh_hi = (qh >> (d * 4 + 16)) & 0xFu;
            r.qs[d    ] |= int32_t((qh_lo * 0x02040810u) & 0x10101010u);
            r.qs[d + 4] |= int32_t((qh_hi * 0x02040810u) & 0x10101010u);
        }
    }
    return r;
}

int32_t get_k_qs_shmem(const uint buf_ib, const uint pos) {
    switch (FaTypeK) {
        case FA_TYPE_Q4_0:
        case FA_TYPE_Q4_1: {
            uint sub = pos % 4;
            uint shift = ((pos % 8) >= 4) ? 4u : 0u;
            return int32_t((uint(kblocksh[buf_ib].qs[sub]) >> shift) & 0x0F0F0F0Fu);
        }
        case FA_TYPE_Q5_0:
        case FA_TYPE_Q5_1: {
            uint sub = pos % 4;
            uint shift = ((pos % 8) >= 4) ? 4u : 0u;
            int32_t result = int32_t((uint(kblocksh[buf_ib].qs[sub]) >> shift) & 0x0F0F0F0Fu);
            uint qh_bits = (kblocksh[buf_ib].qh >> (pos * 4u)) & 0xFu;
            return result | int32_t((qh_bits * 0x02040810u) & 0x10101010u);
        }
        case FA_TYPE_Q8_0: {
            return kblocksh[buf_ib].qs[pos];
        }
        default: return 0;
    }
}

ACC_TYPE k_dot_correction(const uint qib, const ACC_TYPEV2 k_dm) {
    switch (FaTypeK) {
        case FA_TYPE_Q4_0: return -ACC_TYPE(8.0)  * ACC_TYPE(Qf[qib].ds.y) * k_dm.x;
        case FA_TYPE_Q5_0: return -ACC_TYPE(16.0) * ACC_TYPE(Qf[qib].ds.y) * k_dm.x;
        case FA_TYPE_Q4_1:
        case FA_TYPE_Q5_1: return ACC_TYPE(Qf[qib].ds.y) * k_dm.y;
        default: return ACC_TYPE(0.0);
    }
}

void k_block_to_shmem_zero(const uint buf_ib, const uint iqs) {
    kblocksh[buf_ib].qs[iqs] = 0;
    if (iqs == 0) {
        kblocksh[buf_ib].dm = FLOAT_TYPEV2(0.0f);
    }
}
