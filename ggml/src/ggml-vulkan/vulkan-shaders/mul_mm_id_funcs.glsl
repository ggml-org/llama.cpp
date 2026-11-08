#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

void load_row_ids(uint row_begin) {
    for (uint i = gl_LocalInvocationIndex; i < _ne1; i += BLOCK_SIZE) {
        const uint packed_row_id = data_row_map[p.row_ids_offset + row_begin + i];
        row_ids[i] = u16vec2(packed_row_id & 0xffffu, packed_row_id >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
