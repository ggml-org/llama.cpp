// Chunkwise gated delta net for prefill (WY form, as in build_delta_net_chunking), C = 64 tokens:
//   G_i = sum_{j<=i} g_j, D = exp(G_i - G_j), A = strict_lower((K*beta) K^T . D), T = (I + A)^-1
//   W = T (K beta e^G), U = T (V beta), Qg = scale Q e^G, Kg = K e^(G_C - G)
//   V_new = U - W S, O = Qg S + lower(scale Q K^T . D) V_new, S = e^(G_C) S + Kg^T V_new
// kernel_gdn_chunk_prep builds W, U, Qg, Kg, P per (chunk, head, seq); kernel_gdn_chunk_scan walks
// the chunks with NCOL state columns in local memory. S_V = 128 only; the host keeps the rest.

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#ifndef S_V
#define S_V 128
#endif
#ifndef NCOL
#define NCOL 8
#endif

#define CH      64          // tokens per chunk == lanes per workgroup
#define RB      16          // r-block of the A/P phase
#define JB      16          // token block of the A/P phase
#define TSTRIDE 68          // row stride of the A/T triangle in local memory: keeps the T rows float4-aligned

#if S_V != 128
#error "gated_delta_net_chunk.cl is written for S_V == 128"
#endif

// Scratch layout per (chunk, head, seq) index ch, in floats:
//   W, Qg : [S_V/4][CH][4] r-blocked     Kg : [CH/4][S_V][4] t-blocked     U : [CH][S_V] row-major
//   P : [CH/4][CH][4] t-blocked, zero above the diagonal     gamma : exp(G_C), one float
//   Kb, Qb : [S_V/4][CH][4] r-blocked copies of k and q, read only by the prep kernel
#define SCR_W_SIZE   (S_V * CH)
#define SCR_QG_SIZE  (S_V * CH)
#define SCR_KG_SIZE  (S_V * CH)
#define SCR_U_SIZE   (S_V * CH)
#define SCR_P_SIZE   (CH * CH)

__attribute__((reqd_work_group_size(CH, 1, 1)))
REQD_SUBGROUP_SIZE_64
kernel void kernel_gdn_chunk_prep(
        global const char * q_buf,    ulong off_q,
        global const char * k_buf,    ulong off_k,
        global const char * v_buf,    ulong off_v,
        global const char * g_buf,    ulong off_g,
        global const char * beta_buf, ulong off_beta,
        global       char * scr_buf,  ulong off_w, ulong off_qg, ulong off_kg, ulong off_u, ulong off_p, ulong off_gamma,
        ulong off_kb, ulong off_qb,
        uint  H_v,
        uint  H_k,
        uint  hpair,     // 2: this workgroup serves v-heads head and head + H_k (one k-head); 1: one v-head
        uint  n_chunks,
        uint  rq3,
        uint  sq1, uint sq2, uint sq3,
        uint  sk1, uint sk2, uint sk3,
        uint  sv1, uint sv2, uint sv3,
        uint  sb1, uint sb2, uint sb3,
        float scale) {

    global const float * data_q    = (global const float *)(q_buf    + off_q);
    global const float * data_k    = (global const float *)(k_buf    + off_k);
    global const float * data_v    = (global const float *)(v_buf    + off_v);
    global const float * data_g    = (global const float *)(g_buf    + off_g);
    global const float * data_beta = (global const float *)(beta_buf + off_beta);

    const uint chunk = get_group_id(0);
    const uint head  = get_group_id(1);
    const uint seq   = get_group_id(2);
    const uint lane  = get_local_id(0);
    const uint tok   = chunk * CH + lane;

    // the v-heads of this workgroup (hh = 0, 1) and their chunk-head scratch indices
    const uint head1 = head + H_k;
    const uint ch0 = (seq * H_v + head)  * n_chunks + chunk;
    const uint ch1 = (seq * H_v + head1) * n_chunks + chunk;

    global float * scr_kb = (global float *)(scr_buf + off_kb) + (ulong) ch0 * SCR_W_SIZE;
    global float * scr_qb = (global float *)(scr_buf + off_qb) + (ulong) ch0 * SCR_W_SIZE;

    const uint iq1 = head % H_k;
    const uint iq3 = seq / rq3;

    // first row of the chunk in q/k; row j is + j * stride
    global const float * q_chunk = data_q + iq3 * sq3 + iq1 * sq1 + (ulong) (chunk * CH) * sq2;
    global const float * k_chunk = data_k + iq3 * sk3 + iq1 * sk1 + (ulong) (chunk * CH) * sk2;

    __local float AT[CH * TSTRIDE];     // A row-major strictly below the diagonal; T^T on and above it
    __local float A1[CH * (CH - 1) / 2];// head 1's A, packed by row, until its turn
    __local float tile[JB * RB];        // k tile of the A/P product
    __local float Gs[2][CH];            // cumulative gate G_j
    __local float eG[2][CH];            // exp(G_j)
    __local float eGd[2][CH];           // exp(G_C - G_j)
    __local float Bs[2][CH];            // beta_j

    // --- gate prefix, per-token scalars, per head ---------------------------------------------
    float G_i[2], beta_i[2];
    G_i[1] = 0.0f; beta_i[1] = 0.0f;
    if (hpair == 1) {
        // head 1's slots are read (and the values discarded) by the shared loops below
        Gs[1][lane]  = 0.0f;
        eG[1][lane]  = 0.0f;
        eGd[1][lane] = 0.0f;
        Bs[1][lane]  = 0.0f;
    }
    for (uint hh = 0; hh < hpair; hh++) {
        const uint  hd     = (hh == 0) ? head : head1;
        const uint  gb_off = seq * sb3 + hd * sb1 + tok * sb2;
        const float g      = data_g[gb_off];
        const float G      = sub_group_scan_inclusive_add(g);
        const float G_C    = sub_group_broadcast(G, CH - 1);
        G_i[hh]    = G;
        beta_i[hh] = data_beta[gb_off];
        Gs[hh][lane]  = G;
        eG[hh][lane]  = exp(G);
        eGd[hh][lane] = exp(G_C - G);
        Bs[hh][lane]  = beta_i[hh];
        if (lane == 0) {
            *((global float *)(scr_buf + off_gamma) + ((hh == 0) ? ch0 : ch1)) = exp(G_C);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- r-blocked copies Kb, Qb (own-row operands of the A/P product), Qg, and Kg -----------
    // lane = token: one strided row read, coalesced blocked writes
    {
        global const float * q_row = q_chunk + (ulong) lane * sq2;
        global const float * k_row = k_chunk + (ulong) lane * sk2;
        global float * scr_qg0 = (global float *)(scr_buf + off_qg) + (ulong) ch0 * SCR_QG_SIZE;
        global float * scr_qg1 = (global float *)(scr_buf + off_qg) + (ulong) ch1 * SCR_QG_SIZE;
        const float sg0 = scale * eG[0][lane];
        const float sg1 = scale * eG[1][lane];
        #pragma unroll 8
        for (uint r4 = 0; r4 < S_V / 4; r4++) {
            const float4 kv = vload4(0, k_row + r4 * 4);
            const float4 qv = vload4(0, q_row + r4 * 4);
            vstore4(kv,       0, scr_kb  + (r4 * CH + lane) * 4);
            vstore4(qv,       0, scr_qb  + (r4 * CH + lane) * 4);
            vstore4(sg0 * qv, 0, scr_qg0 + (r4 * CH + lane) * 4);
            if (hpair > 1) {
                vstore4(sg1 * qv, 0, scr_qg1 + (r4 * CH + lane) * 4);
            }
        }
    }
    // Kg = e^{G_C - G_t} k_t, t-blocked with lane = r (two rows per lane)
    {
        global float * scr_kg0 = (global float *)(scr_buf + off_kg) + (ulong) ch0 * SCR_KG_SIZE;
        global float * scr_kg1 = (global float *)(scr_buf + off_kg) + (ulong) ch1 * SCR_KG_SIZE;
        #pragma unroll 2
        for (uint rh = 0; rh < S_V / CH; rh++) {
            const uint r = rh * CH + lane;
            #pragma unroll 4
            for (uint t4 = 0; t4 < CH / 4; t4++) {
                float kg0[4], kg1[4];
                #pragma unroll
                for (uint e = 0; e < 4; e++) {
                    const uint  t = t4 * 4 + e;
                    const float k = k_chunk[(ulong) t * sk2 + r];
                    kg0[e] = eGd[0][t] * k;
                    kg1[e] = eGd[1][t] * k;
                }
                vstore4((float4)(kg0[0], kg0[1], kg0[2], kg0[3]), 0, scr_kg0 + (t4 * S_V + r) * 4);
                if (hpair > 1) {
                    vstore4((float4)(kg1[0], kg1[1], kg1[2], kg1[3]), 0, scr_kg1 + (t4 * S_V + r) * 4);
                }
            }
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    // --- A = strict_lower(beta_i D (K K^T)), P = lower(scale D (Q K^T)), lane = row i ---
    // JB tokens j at a time over RB-wide r blocks; the lane's own k and q blocks stay in registers
    // and each wave-uniform float4 of the k tile feeds both dots.
    for (uint jb = 0; jb < CH / JB; jb++) {
        float acc_a[JB], acc_p[JB];
        #pragma unroll
        for (uint j = 0; j < JB; j++) {
            acc_a[j] = 0.0f;
            acc_p[j] = 0.0f;
        }

        for (uint rb = 0; rb < S_V / RB; rb++) {
            float4 ok[RB / 4], oq[RB / 4];
            #pragma unroll
            for (uint r4 = 0; r4 < RB / 4; r4++) {
                ok[r4] = vload4(0, scr_kb + ((rb * (RB / 4) + r4) * CH + lane) * 4);
                oq[r4] = vload4(0, scr_qb + ((rb * (RB / 4) + r4) * CH + lane) * 4);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            {
                // JB tokens x RB floats = 64 lanes x one float4: lane -> (token lane/4, r4 lane%4)
                const uint j  = lane >> 2;
                const uint r4 = lane & 3;
                vstore4(vload4(0, scr_kb + ((rb * (RB / 4) + r4) * CH + jb * JB + j) * 4), 0, tile + j * RB + r4 * 4);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (uint j = 0; j < JB; j++) {
                #pragma unroll
                for (uint r4 = 0; r4 < RB / 4; r4++) {
                    const float4 kj = vload4(0, tile + j * RB + r4 * 4);
                    acc_a[j] += dot(ok[r4], kj);
                    acc_p[j] += dot(oq[r4], kj);
                }
            }
        }

        // per head: A[i][j] = beta_i exp(G_i - G_j) k_i.k_j for j < i (head 1 parked in A1),
        // P[i][j] = scale exp(G_i - G_j) q_i.k_j for j <= i, else 0; t-blocked [j/4][i][4]
        for (uint hh = 0; hh < hpair; hh++) {
            global float * scr_p = (global float *)(scr_buf + off_p) + (ulong) ((hh == 0) ? ch0 : ch1) * SCR_P_SIZE;
            const float Gi = G_i[hh];
            const float bi = beta_i[hh];
            #pragma unroll
            for (uint j = 0; j < JB; j++) {
                const uint jj = jb * JB + j;
                if (jj < lane) {
                    const float a = bi * exp(Gi - Gs[hh][jj]) * acc_a[j];
                    if (hh == 0) {
                        AT[lane * TSTRIDE + jj] = a;
                    } else {
                        A1[lane * (lane - 1) / 2 + jj] = a;
                    }
                }
            }
            #pragma unroll
            for (uint j4 = 0; j4 < JB / 4; j4++) {
                float pe[4];
                #pragma unroll
                for (uint e = 0; e < 4; e++) {
                    const uint jj = jb * JB + j4 * 4 + e;
                    pe[e] = (jj <= lane) ? scale * exp(Gi - Gs[hh][jj]) * acc_p[j4 * 4 + e] : 0.0f;
                }
                vstore4((float4)(pe[0], pe[1], pe[2], pe[3]), 0, scr_p + ((jb * (JB / 4) + j4) * CH + lane) * 4);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint hh = 0; hh < hpair; hh++) {
    if (hh == 1) {
        // head 1's turn: its A into the triangle (own row), then everyone may read it
        for (uint jj = 0; jj < lane; jj++) {
            AT[lane * TSTRIDE + jj] = A1[lane * (lane - 1) / 2 + jj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const uint     ch    = (hh == 0) ? ch0 : ch1;
    global float * scr_w = (global float *)(scr_buf + off_w) + (ulong) ch * SCR_W_SIZE;
    global float * scr_u = (global float *)(scr_buf + off_u) + (ulong) ch * SCR_U_SIZE;
    global const float * v_chunk = data_v + seq * sv3 + ((hh == 0) ? head : head1) * sv1 + (ulong) (chunk * CH) * sv2;

    // --- T = (I + A)^-1 by forward substitution, lane = column l ---
    // T[i][l] = d_il - sum_{j<i} A[i][j] T[j][l]; T[j][l] is this lane's own earlier write, kept at
    // AT[l][j] (on and above the diagonal, disjoint from A), so no barrier is needed.
    if (lane == 0) {
        AT[0] = 1.0f;   // T[0][0]; lane 0 reads it in every later step
    }
    for (uint i = 1; i < CH; i++) {
        float t = (i == lane) ? 1.0f : 0.0f;
        uint j = 0;
        for (; j + 4 <= i; j += 4) {
            const float4 a4 = vload4(0, AT + i * TSTRIDE + j);
            float4 t4 = vload4(0, AT + lane * TSTRIDE + j);
            t4.s0 = (lane <= j)     ? t4.s0 : 0.0f;
            t4.s1 = (lane <= j + 1) ? t4.s1 : 0.0f;
            t4.s2 = (lane <= j + 2) ? t4.s2 : 0.0f;
            t4.s3 = (lane <= j + 3) ? t4.s3 : 0.0f;
            t -= dot(a4, t4);
        }
        for (; j < i; j++) {
            const float tjl = (lane <= j) ? AT[lane * TSTRIDE + j] : 0.0f;
            t -= AT[i * TSTRIDE + j] * tjl;
        }
        if (lane <= i) {
            AT[lane * TSTRIDE + i] = t;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- W = T (K beta e^G), U = T (V beta), lane = column pair (r, r + 64) ---
    // 16-row blocks of i; T for the block is four uniform float4 (AT[j][i0..i0+15]) shared by W and
    // U, and j stops at the block's last row (T is lower triangular).
    for (uint ib = 0; ib < CH / 16; ib++) {
        float w0[16], w1[16], u0[16], u1[16];
        #pragma unroll
        for (uint e = 0; e < 16; e++) {
            w0[e] = 0.0f; w1[e] = 0.0f;
            u0[e] = 0.0f; u1[e] = 0.0f;
        }
        // rows i >= j only: inside the block's own 16 x 16 square the slots [j][i], i < j,
        // hold A, not T, and are masked out
        for (uint j = 0; j < ib * 16 + 16; j++) {
            const float b  = Bs[hh][j];
            const float gk = b * eG[hh][j];
            const float k0 = k_chunk[(ulong) j * sk2 + lane]      * gk;
            const float k1 = k_chunk[(ulong) j * sk2 + CH + lane] * gk;
            const float v0 = v_chunk[(ulong) j * sv2 + lane]      * b;
            const float v1 = v_chunk[(ulong) j * sv2 + CH + lane] * b;
            const int   jd = (int) j - (int) (ib * 16);   // > 0 inside the diagonal square
            #pragma unroll
            for (uint e4 = 0; e4 < 4; e4++) {
                float4 t4 = vload4(0, AT + j * TSTRIDE + ib * 16 + e4 * 4);
                if (jd > 0) {
                    const int bb = (int) (e4 * 4);
                    t4.s0 = (bb + 0 >= jd) ? t4.s0 : 0.0f;
                    t4.s1 = (bb + 1 >= jd) ? t4.s1 : 0.0f;
                    t4.s2 = (bb + 2 >= jd) ? t4.s2 : 0.0f;
                    t4.s3 = (bb + 3 >= jd) ? t4.s3 : 0.0f;
                }
                w0[e4 * 4 + 0] += t4.s0 * k0; w1[e4 * 4 + 0] += t4.s0 * k1; u0[e4 * 4 + 0] += t4.s0 * v0; u1[e4 * 4 + 0] += t4.s0 * v1;
                w0[e4 * 4 + 1] += t4.s1 * k0; w1[e4 * 4 + 1] += t4.s1 * k1; u0[e4 * 4 + 1] += t4.s1 * v0; u1[e4 * 4 + 1] += t4.s1 * v1;
                w0[e4 * 4 + 2] += t4.s2 * k0; w1[e4 * 4 + 2] += t4.s2 * k1; u0[e4 * 4 + 2] += t4.s2 * v0; u1[e4 * 4 + 2] += t4.s2 * v1;
                w0[e4 * 4 + 3] += t4.s3 * k0; w1[e4 * 4 + 3] += t4.s3 * k1; u0[e4 * 4 + 3] += t4.s3 * v0; u1[e4 * 4 + 3] += t4.s3 * v1;
            }
        }
        #pragma unroll
        for (uint e = 0; e < 16; e++) {
            const uint i = ib * 16 + e;
            scr_w[((lane >> 2) * CH + i) * 4 + (lane & 3)]        = w0[e];
            scr_w[(((lane + CH) >> 2) * CH + i) * 4 + (lane & 3)] = w1[e];
            scr_u[i * S_V + lane]      = u0[e];
            scr_u[i * S_V + CH + lane] = u1[e];
        }
    }
    }
}

// One workgroup per (column block, head, seq): NCOL state columns resident in local
// memory across the chunk loop, lane = token row of the chunk and state rows lane,
// lane + 64. Writes the attention rows of the chunked tokens and the state after the
// last chunk (to the snapshot slot, or the tail scratch when the recurrent kernel
// finishes the remaining tokens).
__attribute__((reqd_work_group_size(CH, 1, 1)))
REQD_SUBGROUP_SIZE_64
kernel void kernel_gdn_chunk_scan(
        global const char * scr_buf,  ulong off_w, ulong off_qg, ulong off_kg, ulong off_u, ulong off_p, ulong off_gamma,
        global const char * state_buf, ulong off_state,
        global       char * dst_buf,   ulong off_dst,
        global       char * sout_buf,  ulong off_sout,
        uint  H_v,
        uint  n_tokens,
        uint  n_chunks) {

    const uint cb   = get_group_id(0);
    const uint head = get_group_id(1);
    const uint seq  = get_group_id(2);
    const uint lane = get_local_id(0);

    global const float * data_state = (global const float *)(state_buf + off_state);
    global       float * data_dst   = (global       float *)(dst_buf   + off_dst);
    global       float * data_sout  = (global       float *)(sout_buf  + off_sout);

    const uint state_size = S_V * S_V;
    const uint state_out_base = (seq * H_v + head) * state_size;
    const uint state_in_base  = state_out_base;

    __local float S_l [NCOL * S_V];     // [col][r]
    __local float VN_l[NCOL * CH];      // [col][t]

    const uint col0 = cb * NCOL;

    #pragma unroll
    for (uint c = 0; c < NCOL; c++) {
        S_l[c * S_V + lane]      = data_state[state_in_base + (col0 + c) * S_V + lane];
        S_l[c * S_V + CH + lane] = data_state[state_in_base + (col0 + c) * S_V + CH + lane];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint ch0 = (seq * H_v + head) * n_chunks;
    global float * attn = data_dst + ((ulong) seq * n_tokens * H_v + head) * S_V + col0;

    for (uint chunk = 0; chunk < n_chunks; chunk++) {
        const uint ch = ch0 + chunk;
        global const float * scr_w     = (global const float *)(scr_buf + off_w)     + (ulong) ch * SCR_W_SIZE;
        global const float * scr_qg    = (global const float *)(scr_buf + off_qg)    + (ulong) ch * SCR_QG_SIZE;
        global const float * scr_kg    = (global const float *)(scr_buf + off_kg)    + (ulong) ch * SCR_KG_SIZE;
        global const float * scr_u     = (global const float *)(scr_buf + off_u)     + (ulong) ch * SCR_U_SIZE;
        global const float * scr_p     = (global const float *)(scr_buf + off_p)     + (ulong) ch * SCR_P_SIZE;
        global const float * scr_gamma = (global const float *)(scr_buf + off_gamma) + ch;

        // (V'; Oi) = (W; Qg) S : lane = token row
        float accv[NCOL];
        float acco[NCOL];
        #pragma unroll
        for (uint c = 0; c < NCOL; c++) {
            accv[c] = 0.0f;
            acco[c] = 0.0f;
        }
        for (uint r4 = 0; r4 < S_V / 4; r4++) {
            const float4 w  = vload4(0, scr_w  + (r4 * CH + lane) * 4);
            const float4 qg = vload4(0, scr_qg + (r4 * CH + lane) * 4);
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                const float4 s = vload4(0, S_l + c * S_V + r4 * 4);
                accv[c] += dot(w,  s);
                acco[c] += dot(qg, s);
            }
        }

        // V_new = U - V' into local memory, [col][t]
        #pragma unroll
        for (uint c4 = 0; c4 < NCOL / 4; c4++) {
            const float4 u = vload4(0, scr_u + lane * S_V + col0 + c4 * 4);
            VN_l[(c4 * 4 + 0) * CH + lane] = u.s0 - accv[c4 * 4 + 0];
            VN_l[(c4 * 4 + 1) * CH + lane] = u.s1 - accv[c4 * 4 + 1];
            VN_l[(c4 * 4 + 2) * CH + lane] = u.s2 - accv[c4 * 4 + 2];
            VN_l[(c4 * 4 + 3) * CH + lane] = u.s3 - accv[c4 * 4 + 3];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // O += P V_new (lane = token row); dS = Kg^T V_new (lane = r and r + 64)
        float accs0[NCOL];
        float accs1[NCOL];
        #pragma unroll
        for (uint c = 0; c < NCOL; c++) {
            accs0[c] = 0.0f;
            accs1[c] = 0.0f;
        }
        for (uint t4 = 0; t4 < CH / 4; t4++) {
            const float4 p   = vload4(0, scr_p  + (t4 * CH  + lane) * 4);
            const float4 kg0 = vload4(0, scr_kg + (t4 * S_V + lane) * 4);
            const float4 kg1 = vload4(0, scr_kg + (t4 * S_V + CH + lane) * 4);
            #pragma unroll
            for (uint c = 0; c < NCOL; c++) {
                const float4 vn = vload4(0, VN_l + c * CH + t4 * 4);
                acco [c] += dot(p,   vn);
                accs0[c] += dot(kg0, vn);
                accs1[c] += dot(kg1, vn);
            }
        }

        global float * orow = attn + (ulong) (chunk * CH + lane) * (S_V * H_v);
        #pragma unroll
        for (uint c4 = 0; c4 < NCOL / 4; c4++) {
            vstore4((float4)(acco[c4 * 4 + 0], acco[c4 * 4 + 1], acco[c4 * 4 + 2], acco[c4 * 4 + 3]), 0, orow + c4 * 4);
        }

        // S = gamma S + dS; the barrier also orders this chunk's VN reads before the next writes
        const float gamma = *scr_gamma;
        #pragma unroll
        for (uint c = 0; c < NCOL; c++) {
            S_l[c * S_V + lane]      = gamma * S_l[c * S_V + lane]      + accs0[c];
            S_l[c * S_V + CH + lane] = gamma * S_l[c * S_V + CH + lane] + accs1[c];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    #pragma unroll
    for (uint c = 0; c < NCOL; c++) {
        data_sout[state_out_base + (col0 + c) * S_V + lane]      = S_l[c * S_V + lane];
        data_sout[state_out_base + (col0 + c) * S_V + CH + lane] = S_l[c * S_V + CH + lane];
    }
}
