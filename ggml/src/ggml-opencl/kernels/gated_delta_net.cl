#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifndef S_V
#define S_V 128
#endif
#ifndef KDA
#define KDA 0
#endif
#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 64
#endif
#ifndef LANES_PER_COLUMN
#define LANES_PER_COLUMN 8
#endif
#ifndef COLS_PER_LANE_GROUP
#define COLS_PER_LANE_GROUP 1
#endif
#ifndef SUBGROUPS_PER_WG
#define SUBGROUPS_PER_WG 1
#endif
#ifndef USE_QCOM_SUBGROUP_SHUFFLE
#define USE_QCOM_SUBGROUP_SHUFFLE 0
#endif

#define WG_SIZE             (SUBGROUP_SIZE * SUBGROUPS_PER_WG)
#define LANE_GROUPS_PER_SG  (SUBGROUP_SIZE / LANES_PER_COLUMN)
#define COLS_PER_SG         (LANE_GROUPS_PER_SG * COLS_PER_LANE_GROUP)
#define COLS_PER_WG         (SUBGROUPS_PER_WG * COLS_PER_SG)
#define ROWS_PER_LANE       (S_V / LANES_PER_COLUMN)

#if USE_QCOM_SUBGROUP_SHUFFLE
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#endif

// XOR-based parallel sum
// This does a reduction across groups of LANES_PER_COLUMN
static inline float reduce_add_shmem(float partial, __local float * temp, uint lane) {
#if USE_QCOM_SUBGROUP_SHUFFLE
   #pragma unroll
    for (uint s = LANES_PER_COLUMN / 2u; s > 0u; s >>= 1u) {
        partial += qcom_sub_group_shuffle_xor(partial, s, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, partial);
    }
    return partial;
#else
    temp[lane] = partial;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (uint s = LANES_PER_COLUMN / 2u; s > 0u; s >>= 1u) {
        float other = temp[lane ^ s];
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        temp[lane] += other;
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float result = temp[lane];
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    return result;
#endif
}

#define REDUCE_PARTIAL(partial, temp_ptr, lid) \
    ((LANES_PER_COLUMN == 1u) ? (partial) : reduce_add_shmem((partial), (temp_ptr), (lid)))

// force compiler to optimize kernel for a specific fixed work-group size
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gated_delta_net(
        global const char * q_buf,     ulong off_q,
        global const char * k_buf,     ulong off_k,
        global const char * v_buf,     ulong off_v,
        global const char * g_buf,     ulong off_g,
        global const char * beta_buf,  ulong off_beta,
        global const char * state_buf, ulong off_state,
        global       char * dst_buf,   ulong off_dst,
        uint  H_v,
        uint  n_tokens,
        uint  n_seqs,
        uint  s_off,
        uint  sq1, uint sq2, uint sq3,
        uint  sv1, uint sv2, uint sv3,
        uint  sb1, uint sb2, uint sb3,
        uint  H_k,
        uint  rq3,
        float scale,
        uint K) {

    global const float * data_q     = (global const float *)(q_buf     + off_q);
    global const float * data_k     = (global const float *)(k_buf     + off_k);
    global const float * data_v     = (global const float *)(v_buf     + off_v);
    global const float * data_g     = (global const float *)(g_buf     + off_g);
    global const float * data_beta  = (global const float *)(beta_buf  + off_beta);
    global const float * data_state = (global const float *)(state_buf + off_state);
    global       float * data_dst   = (global       float *)(dst_buf   + off_dst);

    const uint head_id     = get_group_id(0);
    const uint seq_id      = get_group_id(1);
    const uint tid         = (uint)get_local_id(0);

    const uint sg_id       = get_sub_group_id(); // subgroup id
    const uint sg_lid      = get_sub_group_local_id(); // subgroup lane id

    const uint lane        = sg_lid % LANES_PER_COLUMN;
    const uint lane_group  = sg_lid / LANES_PER_COLUMN;
    const uint wg_col_base = get_group_id(2) * COLS_PER_WG;
    const uint sg_col_base = wg_col_base + sg_id * COLS_PER_SG;

    const uint iq1 = head_id % H_k; // head index for Q and K
    const uint iq3 = seq_id / rq3; // seq index for Q and K

    const uint state_size = S_V * S_V;
    // input state holds s0 only [S_v, S_v, H, n_seqs]: per-seq stride is H*D.
    const uint state_base = (seq_id * H_v + head_id) * state_size;
    const uint q_off_base  = iq3 * sq3 + iq1 * sq1;
    const uint v_off_base  = seq_id * sv3 + head_id * sv1;
    const uint gb_off_base = seq_id * sb3 + head_id * sb1;
    const uint state_out_base      = (seq_id * H_v + head_id) * state_size;
    const uint state_size_per_snap = state_size * H_v * n_seqs;

    __local float reduce_temp[WG_SIZE];
    __local float * temp_ptr = reduce_temp + sg_id * SUBGROUP_SIZE;

    float s_shard[COLS_PER_LANE_GROUP][ROWS_PER_LANE];
    #pragma unroll
    for (uint cg = 0; cg < COLS_PER_LANE_GROUP; cg++) {
        const uint col = sg_col_base + cg * LANE_GROUPS_PER_SG + lane_group;
        #pragma unroll
        for (uint r = 0; r < ROWS_PER_LANE; r++) {
            s_shard[cg][r] = data_state[state_base + col * S_V + r * LANES_PER_COLUMN + lane];
        }
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
    uint attn_off = (seq_id * n_tokens * H_v + head_id) * S_V;

    for (uint t = 0; t < n_tokens; t++) {
        const uint  q_off    = q_off_base + t * sq2;
        const uint  k_off    = q_off;
        const uint  v_off    = v_off_base + t * sv2;
        const uint  gb_off   = gb_off_base + t * sb2;
        const float beta_val = data_beta[gb_off];

        float k_reg[ROWS_PER_LANE];
        float q_reg[ROWS_PER_LANE];
#if KDA
        float g_exp[ROWS_PER_LANE];
        #pragma unroll
        for (uint r = 0; r < ROWS_PER_LANE; r++) {
            const uint i = r * LANES_PER_COLUMN + lane;
            k_reg[r] = data_k[k_off + i];
            q_reg[r] = data_q[q_off + i];
            g_exp[r] = exp(data_g[gb_off * S_V + i]);
        }
#else
        const float g_val = exp(data_g[gb_off]);

        #pragma unroll
        for (uint r = 0; r < ROWS_PER_LANE; r++) {
            const uint i = r * LANES_PER_COLUMN + lane;
            k_reg[r] = data_k[k_off + i];
            q_reg[r] = data_q[q_off + i];
        }
#endif

        #pragma unroll
        for (uint cg = 0; cg < COLS_PER_LANE_GROUP; cg++) {
            const uint col = sg_col_base + cg * LANE_GROUPS_PER_SG + lane_group;
            float v_val = data_v[v_off + col];

            float kv_shard = 0.0f;
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_LANE; r++) {
#if KDA
                float gs = g_exp[r] * s_shard[cg][r];
                kv_shard += gs * k_reg[r];
#else
                kv_shard += s_shard[cg][r] * k_reg[r];
#endif
            }

#if !KDA
            kv_shard *= g_val; // Applied once instead of ROWS_PER_LANE times
#endif

            const float kv_col = REDUCE_PARTIAL(kv_shard, temp_ptr, sg_lid);

            const float delta_col = (v_val - kv_col) * beta_val;

            float attn_partial = 0.0f;
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_LANE; r++) {
#if KDA
                float gs = g_exp[r] * s_shard[cg][r];
#else
                float gs = g_val * s_shard[cg][r];
#endif
                s_shard[cg][r] = gs + k_reg[r] * delta_col;
                attn_partial += s_shard[cg][r] * q_reg[r];
            }
            const float attn_col = REDUCE_PARTIAL(attn_partial, temp_ptr, sg_lid);

            if (lane == 0) {
                data_dst[attn_off + col] = attn_col * scale;
            }
        }
        attn_off += S_V * H_v;

        if (K > 1u) {
            const int target_slot = (int)n_tokens - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                #pragma unroll
                for (uint cg = 0; cg < COLS_PER_LANE_GROUP; cg++) {
                    const uint col = sg_col_base + cg * LANE_GROUPS_PER_SG + lane_group;
                    const uint slot_base = s_off + (uint)target_slot * state_size_per_snap + state_out_base;
                    #pragma unroll
                    for (uint r = 0; r < ROWS_PER_LANE; r++) {
                        data_dst[slot_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[cg][r];
                    }
                }
            }
        }
    }

    if (K == 1u) {
        #pragma unroll
        for (uint cg = 0; cg < COLS_PER_LANE_GROUP; cg++) {
            const uint col = sg_col_base + cg * LANE_GROUPS_PER_SG + lane_group;
            #pragma unroll
            for (uint r = 0; r < ROWS_PER_LANE; r++) {
                data_dst[s_off + state_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[cg][r];
            }
        }
    }
}
