module flash_attn_loader;
import types;

public interface IKVLoader {
    static const uint BYTE_SIZE;

    public vector<FLOAT, 4> load(uint element_idx, uint head_dim4_idx);
}

public struct ScalarKVLoader<T: __BuiltinFloatingPointType> : IKVLoader {
    static const uint BYTE_SIZE = sizeof(T) * 4;

    StructuredBuffer<vector<T, 4>> buf;
    uint offset;
    uint stride4;

    public __init(StructuredBuffer<vector<T, 4>> b, uint o, uint s) {
        buf = b;
        offset = o / BYTE_SIZE;
        stride4 = s / 4;
    }

    public vector<FLOAT, 4> load(uint element_idx, uint head_dim4_idx) {
        return vector<FLOAT, 4>(buf[offset + element_idx * stride4 + head_dim4_idx]);
    }
}

public struct Q8_0KVLoader : IKVLoader {
    static const uint BYTE_SIZE = sizeof(block_q8_0_packed16);

    StructuredBuffer<block_q8_0_packed16> buf;
    uint offset;
    uint stride;

    public __init(StructuredBuffer<block_q8_0_packed16> b, uint o, uint s) {
        buf = b;
        offset = o / BYTE_SIZE;
        stride = s * QUANT_K_Q8_0;
    }

    public vector<FLOAT, 4> load(uint element_idx, uint head_dim4_idx) {
        const uint coord = element_idx * stride + 4 * head_dim4_idx;
        const uint ib = coord / QUANT_K_Q8_0;
        const uint iqs = (coord % QUANT_K_Q8_0);

        const vector<FLOAT, 2> v0 = vector<FLOAT, 2>(unpack_s8s32(int32_t(buf[offset + ib].qs[iqs / 2])).xy); // vec4 used due to #12147
        const vector<FLOAT, 2> v1 = vector<FLOAT, 2>(unpack_s8s32(int32_t(buf[offset + ib].qs[iqs / 2 + 1])).xy);

        return FLOAT(buf[offset + ib].d) * vector<FLOAT, 4>(v0.x, v0.y, v1.x, v1.y);
    }
}

public struct Q4_0KVLoader : IKVLoader {
    static const uint BYTE_SIZE = sizeof(block_q4_0_packed16);

    StructuredBuffer<block_q4_0_packed16> buf;
    uint offset;
    uint stride;

    public __init(StructuredBuffer<block_q4_0_packed16> b, uint o, uint s) {
        buf = b;
        offset = o / BYTE_SIZE;
        stride = s * QUANT_K_Q4_0;
    }

    public vector<FLOAT, 4> load(uint element_idx, uint head_dim4_idx) {
        const uint coord = element_idx * stride + 4 * head_dim4_idx;
        const uint ib = coord / QUANT_K_Q4_0;
        const uint iqs = (coord % QUANT_K_Q4_0);

        uint vui_lo = uint(buf[offset + ib].qs[(iqs & 0xF) / 2 + 0]);
        uint vui_hi = uint(buf[offset + ib].qs[(iqs & 0xF) / 2 + 1]);
        const uint shift = (iqs & 0x10) >> 2;
        vui_lo >>= shift;
        vui_hi >>= shift;

        return FLOAT(buf[offset + ib].d) * (vector<FLOAT, 4>(FLOAT(vui_lo & 0xF), FLOAT((vui_lo >> 8) & 0xF),
                                                             FLOAT(vui_hi & 0xF), FLOAT((vui_hi >> 8) & 0xF)) - FLOAT(8.0f));
    }
}
