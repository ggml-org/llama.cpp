#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
int32_t get_k_qs(uint ib, uint iqs, uint a_offset) {
#ifdef DATA_A_Q4_0
    uint vui = pack32(u16vec2(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0],
                              k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]));
#else
    uint vui = k_packed32.k_data_packed32[a_offset + ib].qs[(iqs & 0xF) / 4];
#endif

    uint shift = (iqs & 0x10) >> 2;
    vui >>= shift;

    return int32_t(vui & 0x0F0F0F0F);
}
#endif

#if defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
int32_t get_k_qs(uint ib, uint iqs, uint a_offset) {
#ifdef DATA_A_Q5_0
    uint vui = pack32(u16vec2(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0],
                              k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]));
    uint qh = pack32(u16vec2(k_packed.k_data_packed16[a_offset + ib].qh[0],
                             k_packed.k_data_packed16[a_offset + ib].qh[1]));
#else
    uint vui = k_packed32.k_data_packed32[a_offset + ib].qs[(iqs & 0xF) / 4];
    uint qh = k_packed.k_data_packed16[a_offset + ib].qh;
#endif

    uint shift = (iqs & 0x10) >> 2;
    vui >>= shift;

    uint qh_bits = (qh >> iqs) & 0xF;
    return int32_t(vui & 0x0F0F0F0F) | int32_t((qh_bits * 0x02040810u) & 0x10101010u);
}
#endif

#if defined(DATA_A_Q8_0)
int32_t get_k_qs(uint ib, uint iqs, uint a_offset) {
    return pack32(i16vec2(k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2], k_packed.k_data_packed16[a_offset + ib].qs[iqs / 2 + 1]));
}
#endif

#if defined(DATA_A_IQ4_NL)
int32_t get_k_qs(uint ib, uint iqs, uint a_offset) {
    uint vui = pack32(u16vec2(k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 0],
                              k_packed.k_data_packed16[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]));
    uint shift = (iqs & 0x10) >> 2;
    vui >>= shift;

    u8vec4 idx = unpack8(vui & 0x0F0F0F0F);
    return pack32(i8vec4(kvalues_iq4nl_const[idx.x],
                         kvalues_iq4nl_const[idx.y],
                         kvalues_iq4nl_const[idx.z],
                         kvalues_iq4nl_const[idx.w]));
}
#endif

#if QUANT_AUXF == 1
FLOAT_TYPE get_k_d(uint ib, uint a_offset) {
    return FLOAT_TYPE(k_packed.k_data_packed16[a_offset + ib].d);
}
#else
FLOAT_TYPEV2 get_k_dm(uint ib, uint a_offset) {
    return FLOAT_TYPEV2(k_packed32.k_data_packed32[a_offset + ib].dm);
}
#endif

void k_block_to_shmem(const uint buf_ib, const uint global_ib, const uint iqs, const uint a_offset) {
#if defined(DATA_A_Q4_0)
    kblocksh[buf_ib].qs[iqs] = pack32(u16vec2(k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2],
                                              k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2 + 1]));
#elif defined(DATA_A_Q4_1)
    kblocksh[buf_ib].qs[iqs] = k_packed32.k_data_packed32[a_offset + global_ib].qs[iqs];
#elif defined(DATA_A_Q5_0)
    kblocksh[buf_ib].qs[iqs] = pack32(u16vec2(k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2],
                                              k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2 + 1]));
    if (iqs == 0) {
        kblocksh[buf_ib].qh = pack32(u16vec2(k_packed.k_data_packed16[a_offset + global_ib].qh[0],
                                             k_packed.k_data_packed16[a_offset + global_ib].qh[1]));
    }
#elif defined(DATA_A_Q5_1)
    kblocksh[buf_ib].qs[iqs] = k_packed32.k_data_packed32[a_offset + global_ib].qs[iqs];
    if (iqs == 0) {
        kblocksh[buf_ib].qh = k_packed.k_data_packed16[a_offset + global_ib].qh;
    }
#elif defined(DATA_A_Q8_0)
    kblocksh[buf_ib].qs[iqs] = pack32(i16vec2(k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2],
                                              k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2 + 1]));
#elif defined(DATA_A_IQ4_NL)
    const uint qs = pack32(u16vec2(k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2],
                                   k_packed.k_data_packed16[a_offset + global_ib].qs[iqs * 2 + 1]));
    const u8vec4 i_a0 = unpack8( qs       & 0x0F0F0F0F);
    const u8vec4 i_a1 = unpack8((qs >> 4) & 0x0F0F0F0F);
    kblocksh[buf_ib].qs[iqs    ] = pack32(i8vec4(kvalues_iq4nl_const[i_a0.x], kvalues_iq4nl_const[i_a0.y],
                                                 kvalues_iq4nl_const[i_a0.z], kvalues_iq4nl_const[i_a0.w]));
    kblocksh[buf_ib].qs[iqs + 4] = pack32(i8vec4(kvalues_iq4nl_const[i_a1.x], kvalues_iq4nl_const[i_a1.y],
                                                 kvalues_iq4nl_const[i_a1.z], kvalues_iq4nl_const[i_a1.w]));
#endif

    if (iqs == 0) {
#if QUANT_AUXF == 1
        kblocksh[buf_ib].dm = FLOAT_TYPE(k_packed.k_data_packed16[a_offset + global_ib].d);
#else
        kblocksh[buf_ib].dm = FLOAT_TYPEV2(k_packed32.k_data_packed32[a_offset + global_ib].dm);
#endif
    }
}

int32_t get_k_qs_shmem(const uint buf_ib, const uint pos) {
#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
    uint sub = pos % 4;
    uint shift = ((pos % 8) >= 4) ? 4 : 0;
    return int32_t((kblocksh[buf_ib].qs[sub] >> shift) & 0x0F0F0F0F);
#elif defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
    uint sub = pos % 4;
    uint shift = ((pos % 8) >= 4) ? 4 : 0;
    int32_t result = int32_t((kblocksh[buf_ib].qs[sub] >> shift) & 0x0F0F0F0F);
    uint qh_bits = (kblocksh[buf_ib].qh >> (pos * 4)) & 0xF;
    return result | int32_t((qh_bits * 0x02040810u) & 0x10101010u);
#elif defined(DATA_A_Q8_0) || defined(DATA_A_IQ4_NL)
    return kblocksh[buf_ib].qs[pos];
#endif
}

ACC_TYPE k_dot_correction(const uint qib, const ACC_TYPEV2 k_dm) {
#if defined(DATA_A_Q4_0)
    return -ACC_TYPE(8.0) * ACC_TYPE(Qf[qib].ds.y) * k_dm.x;
#elif defined(DATA_A_Q5_0)
    return -ACC_TYPE(16.0) * ACC_TYPE(Qf[qib].ds.y) * k_dm.x;
#elif defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
    return ACC_TYPE(Qf[qib].ds.y) * k_dm.y;
#else
    return ACC_TYPE(0.0);
#endif
}

void k_block_to_shmem_zero(const uint buf_ib, const uint iqs) {
    kblocksh[buf_ib].qs[iqs] = 0;
#if defined(DATA_A_IQ4_NL)
    kblocksh[buf_ib].qs[iqs + 4] = 0;
#endif
    if (iqs == 0) {
#if QUANT_AUXF == 1
        kblocksh[buf_ib].dm = FLOAT_TYPE(0.0f);
#else
        kblocksh[buf_ib].dm = FLOAT_TYPEV2(0.0f);
#endif
    }
}
