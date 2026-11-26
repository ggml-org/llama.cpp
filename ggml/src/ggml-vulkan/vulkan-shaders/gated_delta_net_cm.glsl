layout(push_constant) uniform Parameters {
    uint H;
    uint n_tokens;
    uint n_seqs;
    uint s_off;
    uint sq1, sq2, sq3;
    uint sk1, sk2, sk3;
    uint sv1, sv2, sv3;
    uint sb1, sb2, sb3;
    uint neq1, rq3;
    float scale;
    uint K;
#ifdef GDN_REPLAY
    uint n_pend, keep;
    uint spk1, spk2, spk3;
    uint spv1, spv2, spv3;
    uint spg1, spg2, spg3;
#endif
} p;

layout(binding = 0) readonly  buffer QBuf     { float data_q[];     };
layout(binding = 1) readonly  buffer KBuf     { float data_k[];     };
layout(binding = 2) readonly  buffer VBuf     { float data_v[];     };
layout(binding = 3) readonly  buffer GBuf     { float data_g[];     };
layout(binding = 4) readonly  buffer BetaBuf  { float data_beta[];  };
layout(binding = 5) readonly  buffer StateBuf { float data_state[]; };
layout(binding = 6)           buffer DstBuf   { float data_dst[];   };

#ifdef GDN_REPLAY
layout(binding = 7)  readonly buffer PKBuf   { float data_pk[];    };
layout(binding = 8)  readonly buffer PVBuf   { float data_pv[];    };
layout(binding = 9)  readonly buffer PGBBuf  { float data_pgb[];   };
layout(binding = 10) readonly buffer NRepBuf { int   data_n_rep[]; };

uint replay_count(uint seq_id) {
    return uint(clamp(data_n_rep[seq_id], 0, int(p.n_pend)));
}

void replay_inputs(uint seq_id, uint head_id, uint iq1, uint t,
                   out uint k_off, out uint v_off, out float decay, out float beta) {
    k_off = seq_id * p.spk3 + t * p.spk2 + iq1 * p.spk1;
    v_off = seq_id * p.spv3 + t * p.spv2 + head_id * p.spv1;
    const uint g_off = seq_id * p.spg3 + t * p.spg2 + head_id * p.spg1;
    decay = exp(data_pgb[g_off]);
    beta = data_pgb[g_off + p.n_pend * p.spg2];
}

float replay_sum(const in float a, const in float b) {
    return a + b;
}
#endif
