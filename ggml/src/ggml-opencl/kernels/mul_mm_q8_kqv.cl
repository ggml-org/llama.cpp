#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// int8 KQV for the decomposed prefill: out[d, q, h] = sum_kv V^T[d, kv, h] * P[kv, q, h].
// P (u8, kernel_soft_max_4_f16_q8) and V^T (s8, flash_attn_v_transpose_q8) are symmetric with one
// scale per 32 along kv, so a block is 8 uints fed to dp4a. One lane owns one d row and all
// KQV_TN queries of the tile; P is read from local memory. Dispatch order (matched with the host):
// d-block, GQA-group heads, query tile, KV head, so a group's heads share each V^T slice in L2.
// The packed dot takes the V word (registers) first: swapped operands miscompile on Adreno.

#define KQV_TM 64   // d rows per workgroup, one per lane
#ifndef KQV_TN
#define KQV_TN 32   // queries per workgroup, all owned by every lane
#endif
#ifndef KQV_NB
#define KQV_NB 8    // 32-blocks of P staged per barrier
#endif
#define KQV_WG 64

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_q8_kqv(
        global const uint  * vq,        // V^T int8, [head_kv][dv][kv_pitch]     as packed uints
        ulong                off_vq,
        global const half  * vd,        // V^T scales, [head_kv][dv][kv_pitch/32]
        ulong                off_vd,
        global const uint  * pq,        // P u8,     [head][n_q][kv_pitch]       as packed uints
        ulong                off_pq,
        global const float * pd,        // P scales f32, [head][n_q][kv_pitch/32]
        ulong                off_pd,
        global       float * dst,       // [head][n_q][dv]
        ulong                off_dst,
        int                  dv,
        int                  n_kv,
        int                  n_q,
        int                  n_head,
        int                  n_head_kv,
        int                  kv_pitch    // bytes per V^T row and per P row, >= n_kv, multiple of 32
) {
    vq  = (global const uint  *)((global const char *)vq  + off_vq);
    vd  = (global const half  *)((global const char *)vd  + off_vd);
    pq  = (global const uint  *)((global const char *)pq  + off_pq);
    pd  = (global const float *)((global const char *)pd  + off_pd);
    dst = (global       float *)((global const char *)dst + off_dst);

    const int lid  = get_local_id(0);
    const int gsz  = n_head / n_head_kv;            // query heads per KV head
    const int g1   = get_group_id(1);
    const int head_kv = get_group_id(2);
    const int head = head_kv*gsz + (g1 % gsz);
    const int qn0  = (g1 / gsz) * KQV_TN;           // first query of this tile
    const int d    = get_group_id(0)*KQV_TM + lid;  // this lane's d row

    const int nblk = n_kv / 32;                     // dispatch guarantees n_kv % 32 == 0
    const int nu   = kv_pitch / 4;                  // uints per row
    const int nbp  = kv_pitch / 32;                 // scales per row

    __local uint  sh_pq[KQV_TN][KQV_NB][8];
    __local float sh_pd[KQV_TN][KQV_NB];

    float acc[KQV_TN];
    #pragma unroll
    for (int t = 0; t < KQV_TN; ++t) {
        acc[t] = 0.0f;
    }

    const int    dl    = min(d, dv - 1);            // clamp so a tail lane loads in bounds
    const size_t vbase = ((size_t)head_kv*dv + dl)*nu;
    const size_t vdbas = ((size_t)head_kv*dv + dl)*nbp;
    const size_t pbase = (size_t)head*n_q;

    for (int bg = 0; bg < nblk; bg += KQV_NB) {
        const int nb_here = min(KQV_NB, nblk - bg);

        // stage P: each (query, block) is 8 uints = two 16-byte loads
        for (int i = lid; i < KQV_TN*KQV_NB*2; i += KQV_WG) {
            const int t  = i / (KQV_NB*2);
            const int bb = (i / 2) % KQV_NB;
            const int h  = (i & 1) * 4;
            const int qi = qn0 + t;
            uint4 v = (uint4)(0u);
            if (qi < n_q && bb < nb_here) {
                v = vload4(0, &pq[(pbase + qi)*nu + (size_t)(bg + bb)*8 + h]);
            }
            vstore4(v, 0, &sh_pq[t][bb][h]);
        }
        for (int i = lid; i < KQV_TN*KQV_NB; i += KQV_WG) {
            const int t  = i / KQV_NB;
            const int bb = i % KQV_NB;
            const int qi = qn0 + t;
            sh_pd[t][bb] = (qi < n_q && bb < nb_here)
                ? pd[(pbase + qi)*nbp + bg + bb] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int bb = 0; bb < nb_here; ++bb) {
            const uint4 w0 = vload4(0, &vq[vbase + (size_t)(bg + bb)*8]);
            const uint4 w1 = vload4(0, &vq[vbase + (size_t)(bg + bb)*8 + 4]);
            const float dvs = (float)vd[vdbas + bg + bb];

            #pragma unroll
            for (int t = 0; t < KQV_TN; ++t) {
                const uint4 a0 = vload4(0, &sh_pq[t][bb][0]);
                const uint4 a1 = vload4(0, &sh_pq[t][bb][4]);
                int raw = 0;
                raw = dot_acc_sat_4x8packed_su_int(w0.s0, a0.s0, raw);
                raw = dot_acc_sat_4x8packed_su_int(w0.s1, a0.s1, raw);
                raw = dot_acc_sat_4x8packed_su_int(w0.s2, a0.s2, raw);
                raw = dot_acc_sat_4x8packed_su_int(w0.s3, a0.s3, raw);
                raw = dot_acc_sat_4x8packed_su_int(w1.s0, a1.s0, raw);
                raw = dot_acc_sat_4x8packed_su_int(w1.s1, a1.s1, raw);
                raw = dot_acc_sat_4x8packed_su_int(w1.s2, a1.s2, raw);
                raw = dot_acc_sat_4x8packed_su_int(w1.s3, a1.s3, raw);
                acc[t] += dvs * sh_pd[t][bb] * (float)raw;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (d >= dv) {
        return;
    }

    // dst is [head][n_q][dv] with dv contiguous, so the 64 lanes of a d-block write contiguously
    #pragma unroll
    for (int t = 0; t < KQV_TN; ++t) {
        const int qi = qn0 + t;
        if (qi < n_q) {
            dst[(pbase + qi)*dv + d] = acc[t];
        }
    }
}
