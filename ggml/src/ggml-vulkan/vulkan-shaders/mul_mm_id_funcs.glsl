#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

#ifdef MUL_MAT_ID_USE_SUBGROUPS
shared uvec4 ballots_sh[NUM_WARPS];

void load_row_ids(uint expert_idx, bool nei0_is_pow2, uint ic) {
    _ne1 = 0;
    uint num_elements = p.nei1 * p.nei0;
    uint nei0shift = findLSB(p.nei0);

    uint ids[16];
    uint iter = 0;

    uint expert_count = data_expert_count[expert_idx];

    for (uint j = 0; j < num_elements; j += BLOCK_SIZE) {
        // prefetch up to 16 elements
        if (iter == 0) {
            [[unroll]] for (uint k = 0; k < 16; ++k) {
                uint i = j + gl_LocalInvocationIndex + k*BLOCK_SIZE;
                bool in_range = i < num_elements;
                uint ii1;
                if (nei0_is_pow2) {
                    ii1 = i >> nei0shift;
                } else {
                    ii1 = i / p.nei0;
                }
                uint ii0 = i - ii1 * p.nei0;
                ids[k] = in_range ? data_ids[ii1*p.nbi1 + ii0] : 0;
            }
        }
        uint i = j + gl_LocalInvocationIndex;
        bool in_range = i < num_elements;
        uint ii1;
        if (nei0_is_pow2) {
            ii1 = i >> nei0shift;
        } else {
            ii1 = i / p.nei0;
        }
        uint ii0 = i - ii1 * p.nei0;
        uint id = ids[iter++];
        uvec4 ballot = subgroupBallot(in_range && id == expert_idx);

        if (gl_SubgroupInvocationID == 0) {
            ballots_sh[gl_SubgroupID] = ballot;
        }
        barrier();

        uint subgroup_base = 0;
        uint total = 0;
        for (uint k = 0; k < gl_NumSubgroups; ++k) {
            if (k == gl_SubgroupID) {
                subgroup_base = total;
            }
            total += subgroupBallotBitCount(ballots_sh[k]);
        }
        barrier();

        uint idx = subgroup_base + subgroupBallotExclusiveBitCount(ballot);
        if (in_range && id == expert_idx && _ne1 + idx >= ic * BN && _ne1 + idx < (ic + 1) * BN) {
            row_ids[_ne1 + idx - ic * BN] = u16vec2(ii0, ii1);
        }
        _ne1 += total;
        iter &= 15;
        if (_ne1 >= (ic + 1) * BN || _ne1 == expert_count) {
            break;
        }
    }
    barrier();
}
#endif // MUL_MAT_ID_USE_SUBGROUPS

void load_row_ids_hoisted(uint expert_idx, uint ic) {
    _ne1 = uint(data_expert_count[expert_idx]);

    const uint tile_begin = ic * BN;
    const uint tile_count = tile_begin < _ne1 ? min(BN, _ne1 - tile_begin) : 0;
    const uint expert_offset = uint(data_expert_count[p.n_experts + expert_idx]);
    const uint row_ids_offset = 2 * p.n_experts + 1 + expert_offset + tile_begin;

    for (uint i = gl_LocalInvocationIndex; i < tile_count; i += BLOCK_SIZE) {
        const uint packed_row_id = uint(data_expert_count[row_ids_offset + i]);
        row_ids[i] = u16vec2(packed_row_id & 0xffffu, packed_row_id >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
