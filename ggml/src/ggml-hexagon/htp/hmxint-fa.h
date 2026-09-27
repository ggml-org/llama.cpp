// hmxint-fa: integer-HMX flash attention (prefill) for SoCs without FP16 HMX (e.g. SM7750).
// O = softmax(scale * Q K^T + mask) V, per kv head:
//   K channels are smoothed (divided by their max over keys, Q multiplied back) so one scale per key is accurate;
//   K -> int8 "weights" (columns = keys, k = head dim), V^T -> int8 "weights" (columns = channels, k = keys);
//   S = Q' K'^T and O = P V on HMX (a16 x w8, coarse + fine stores, see hmxint-core.h); softmax on HVX,
//   the 1/rowsum normalisation is folded into the P row scale.
// Supported: q f32, k/v f16, dst f32, no sinks / softcap / ALiBi. Otherwise HTP_STATUS_NO_SUPPORT (caller falls back).

#include "hmxint-core.h"
#include <HAP_perf.h>

#define HMXFA_MAX_KV   8192
#define HMXFA_MAX_D    512

struct hmxfa_w { uint8_t * tiles; uint32_t * bias; float * k512; float * C; int nct, nkt, ng, N; };

struct hmxfa_state {
    struct htp_context * ctx;
    const struct htp_tensor *q, *k, *v, *mask, *dst;
    int d, dv, n_q, n_kv, n_kv_pad, n_head, n_head_kv, G, seq, hk, h;
    float scale;
    int n_threads;
    float sd[HMXFA_MAX_D], isd[HMXFA_MAX_D];
    float spart[8][HMXFA_MAX_D];
    struct hmxfa_w W1, W2;
    // current row chunk of head h: rows i0 .. i0 + nr - 1
    int i0, nr, nrt;
    float * qs;          // [nr][d]        scaled Q rows (DDR)
    float * S;           // [nr][n_kv_pad] scores / probabilities (DDR)
    float sa[1024 + 16], ia[1024 + 16];
    // engine
    const float * act; int act_stride, K;     // activation rows for the current matmul
    struct hmxfa_w * W;
    int out_is_S;                             // 1: rows -> S, 0: rows -> dst
    uint8_t * v_act; uint8_t * v_stage;
    int8_t *  k8;        // [n_kv_pad][d] int8 K rows (DDR)
    int32_t * kS;        // [n_kv_pad] per-key sum of int8
    uint16_t * kS0;      // [n_kv_pad] per-key fp16 scale bits
    float *   vf;        // [n_kv_pad][dv] V rows as f32 (DDR)
    float     vmax_part[8][HMXFA_MAX_D];
    float     sV[HMXFA_MAX_D], iV[HMXFA_MAX_D]; uint16_t sVb[HMXFA_MAX_D];
    uint64_t t[12];
};

static float hmxfa_k512_1[HMXFA_MAX_KV], hmxfa_C_1[HMXFA_MAX_KV], hmxfa_k512_2[HMXFA_MAX_D], hmxfa_C_2[HMXFA_MAX_D];
static float * hmxfa_qs_buf = NULL; static size_t hmxfa_qs_sz = 0;
static float * hmxfa_S_buf  = NULL; static size_t hmxfa_S_sz  = 0;
static uint8_t * hmxfa_kv_buf = NULL; static size_t hmxfa_kv_sz = 0;

static inline const uint16_t * hmxfa_krow(struct hmxfa_state * st, int j) {
    return (const uint16_t *) (st->k->data + (size_t) j * st->k->nb[1] + (size_t) st->hk * st->k->nb[2] + (size_t) st->seq * st->k->nb[3]);
}
static inline const uint16_t * hmxfa_vrow(struct hmxfa_state * st, int j) {
    return (const uint16_t *) (st->v->data + (size_t) j * st->v->nb[1] + (size_t) st->hk * st->v->nb[2] + (size_t) st->seq * st->v->nb[3]);
}


static HVX_VectorPair hmxfa_h2f_vec(HVX_Vector v);

// round-half-away f32 -> int32
static inline HVX_Vector hmxfa_round_w(HVX_Vector x) {
    HVX_Vector half = Q6_V_vor_VV(Q6_V_vsplat_R(0x3f000000), Q6_V_vand_VV(x, Q6_V_vsplat_R(0x80000000)));
    return Q6_Vw_equals_Vsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, half)));
}
// 32 int32 -> 32 int8 in the low 32 bytes (saturating)
static inline HVX_Vector hmxfa_w2b(HVX_Vector w) {
    HVX_Vector h = Q6_Vh_vpack_VwVw_sat(Q6_V_vzero(), w);
    return Q6_Vb_vpack_VhVh_sat(Q6_V_vzero(), h);
}

// channel max of |K| over keys (per worker partial)
static void hmxfa_sd_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data;
    const HVX_Vector absm = Q6_V_vsplat_R(0x7fffffff);
    HVX_Vector m[HMXFA_MAX_D / 32];
    for (int c = 0; c < st->d / 32; c++) m[c] = Q6_V_vzero();
    for (int j = i; j < st->n_kv; j += n) {
        const uint16_t * kr = hmxfa_krow(st, j);
        for (int c = 0; c < st->d; c += 64) {
            HVX_VectorPair f = hmxfa_h2f_vec(hmxi_ldu(kr + c));
            m[c / 32] = Q6_Vw_vmax_VwVw(m[c / 32], Q6_V_vand_VV(Q6_V_lo_W(f), absm));
            if (c + 32 < st->d) m[c / 32 + 1] = Q6_Vw_vmax_VwVw(m[c / 32 + 1], Q6_V_vand_VV(Q6_V_hi_W(f), absm));
        }
    }
    for (int c = 0; c < st->d / 32; c++) hmxi_stu(st->spart[i] + c * 32, m[c]);
}

// per key row: K' = K * isd -> int8 row + per-key scale/sum; V row -> f32 + per-channel max (worker partial)
static void hmxfa_rows_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data;
    const HVX_Vector absm = Q6_V_vsplat_R(0x7fffffff), ones = Q6_Vb_vsplat_R(1);
    HVX_Vector vm[HMXFA_MAX_D / 32];
    for (int c = 0; c < st->dv / 32; c++) vm[c] = Q6_V_vzero();
    HVX_Vector kf[HMXFA_MAX_D / 32];
    for (int j = i; j < st->n_kv_pad; j += n) {
        int8_t * k8 = st->k8 + (size_t) j * st->d;
        if (j >= st->n_kv) { memset(k8, 0, st->d); st->kS[j] = 0; st->kS0[j] = 0; memset(st->vf + (size_t) j * st->dv, 0, st->dv * 4); continue; }
        const uint16_t * kr = hmxfa_krow(st, j);
        HVX_Vector am = Q6_V_vzero();
        for (int c = 0; c < st->d; c += 64) {
            HVX_VectorPair f = hmxfa_h2f_vec(hmxi_ldu(kr + c));
            kf[c / 32] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(f), hmxi_ldu(st->isd + c)));
            am = Q6_Vw_vmax_VwVw(am, Q6_V_vand_VV(kf[c / 32], absm));
            if (c + 32 < st->d) {
                kf[c / 32 + 1] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(f), hmxi_ldu(st->isd + c + 32)));
                am = Q6_Vw_vmax_VwVw(am, Q6_V_vand_VV(kf[c / 32 + 1], absm));
            }
        }
        int32_t amv[32]; hmxi_stu(amv, am); int32_t mx = 0; for (int t = 0; t < 32; t++) mx = amv[t] > mx ? amv[t] : mx;
        float amf; memcpy(&amf, &mx, 4);
        int32_t S = 0;
        if (amf > 0.0f) {
            uint16_t sb = hmxi_f16_round_up_bits(amf / 127.0f); st->kS0[j] = sb;
            const HVX_Vector vinv = Q6_V_vsplat_R(hmxi_f2u(1.0f / hmxi_h2f(sb)));
            HVX_Vector acc = Q6_V_vzero();
            for (int c = 0; c < st->d; c += 32) {
                HVX_Vector b = hmxfa_w2b(hmxfa_round_w(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(kf[c / 32], vinv))));
                memcpy(k8 + c, &b, 32);
                acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vrmpy_VbVb(b, ones));   // low 8 words hold the sums of this 32-chunk
            }
            int32_t sv[32]; hmxi_stu(sv, acc); for (int t = 0; t < 8; t++) S += sv[t];
        } else { memset(k8, 0, st->d); st->kS0[j] = 0; }
        st->kS[j] = S;
        // V row
        const uint16_t * vr = hmxfa_vrow(st, j);
        float * vo = st->vf + (size_t) j * st->dv;
        for (int c = 0; c < st->dv; c += 64) {
            HVX_VectorPair f = hmxfa_h2f_vec(hmxi_ldu(vr + c));
            hmxi_stu(vo + c, Q6_V_lo_W(f)); vm[c / 32] = Q6_Vw_vmax_VwVw(vm[c / 32], Q6_V_vand_VV(Q6_V_lo_W(f), absm));
            if (c + 32 < st->dv) { hmxi_stu(vo + c + 32, Q6_V_hi_W(f)); vm[c / 32 + 1] = Q6_Vw_vmax_VwVw(vm[c / 32 + 1], Q6_V_vand_VV(Q6_V_hi_W(f), absm)); }
        }
    }
    for (int c = 0; c < st->dv / 32; c++) hmxi_stu(st->vmax_part[i] + c * 32, vm[c]);
}

static inline void hmxfa_bias(uint32_t * b, const uint16_t * s0b, const int32_t * S) {
    const float lim = 16777216.0f / (32.0f * HMXI_G * 65535.0f * 127.0f);
    for (int c = 0; c < 32; c++) {
        uint16_t co = 0;
        if (s0b[c]) {
            int ex; frexpf(lim / hmxi_h2f(s0b[c]), &ex); int jj = ex - 1;
            int e = ((s0b[c] >> 10) & 31) + jj;
            co = e < 0 ? 0 : (uint16_t) ((e << 10) | (s0b[c] & 1023));
        }
        uint16_t fi = co ? (uint16_t) (co + (8 << 10)) : 0;
        uint32_t b2 = (uint32_t) (-128 * S[c]);
        b[c] = co; b[32 + c] = b2; b[64 + c] = fi; b[96 + c] = b2;
    }
}
static inline float hmxfa_k512(uint16_t s0b) {
    if (!s0b) return 0.0f;
    const float lim = 16777216.0f / (32.0f * HMXI_G * 65535.0f * 127.0f);
    int ex; frexpf(lim / hmxi_h2f(s0b), &ex); return ldexpf(2.0f, -(ex - 1));
}

// W1: columns = keys, k = d. Tile row r of k-tile kt: word per key = K8[key][kt*32 + r*4 .. +3]
static void hmxfa_w1_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data; struct hmxfa_w * W = &st->W1;
    for (int ct = i; ct < W->nct; ct += n) {
        for (int kt = 0; kt < W->nkt; kt++) {
            uint32_t * t = (uint32_t *) (W->tiles + ((size_t) ct * W->nkt + kt) * 1024);
            for (int r = 0; r < 8; r++) for (int key = 0; key < 32; key++)
                t[r * 32 + key] = *(const uint32_t *) (st->k8 + (size_t) (ct * 32 + key) * st->d + kt * 32 + r * 4);
        }
        for (int c = 0; c < 32; c++) { W->k512[ct * 32 + c] = hmxfa_k512(st->kS0[ct * 32 + c]); W->C[ct * 32 + c] = -0.5f * W->ng; }
        for (int g = 0; g < W->ng; g++) hmxfa_bias(W->bias + ((size_t) ct * W->ng + g) * 128, st->kS0 + ct * 32, st->kS + ct * 32);
    }
}

// W2: columns = channels, k = keys. For 4 keys: bytes c*4 + t = V8[4q+t][c]
static void hmxfa_w2_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data; struct hmxfa_w * W = &st->W2;
    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    for (int ct = i; ct < W->nct; ct += n) {
        const HVX_Vector vinv = hmxi_ldu(st->iV + ct * 32);
        for (int g = 0; g < W->ng; g++) {
            HVX_Vector S = Q6_V_vzero();
            int kt0 = g * HMXI_G, nb = W->nkt - kt0 < HMXI_G ? W->nkt - kt0 : HMXI_G;
            for (int kt = kt0; kt < kt0 + nb; kt++) {
                uint8_t * t = W->tiles + ((size_t) ct * W->nkt + kt) * 1024;
                for (int q = 0; q < 8; q++) {
                    HVX_Vector b[4];
                    for (int u = 0; u < 4; u++) {
                        const float * vr = st->vf + (size_t) (kt * 32 + q * 4 + u) * st->dv + ct * 32;
                        b[u] = hmxfa_w2b(hmxfa_round_w(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hmxi_ldu(vr), vinv))));
                    }
                    HVX_Vector A = Q6_V_lo_W(Q6_W_vshuff_VVR(b[1], b[0], -1));
                    HVX_Vector B = Q6_V_lo_W(Q6_W_vshuff_VVR(b[3], b[2], -1));
                    HVX_Vector R = Q6_V_lo_W(Q6_W_vshuff_VVR(B, A, -2));
                    *(HVX_Vector *) (t + q * 128) = R;
                    S = Q6_Vw_vadd_VwVw(S, Q6_Vw_vrmpy_VbVb(R, ones));
                }
            }
            int32_t Sv[32]; hmxi_stu(Sv, S);
            hmxfa_bias(W->bias + ((size_t) ct * W->ng + g) * 128, st->sVb + ct * 32, Sv);
        }
        for (int c = 0; c < 32; c++) { W->k512[ct * 32 + c] = hmxfa_k512(st->sVb[ct * 32 + c]); W->C[ct * 32 + c] = -0.5f * W->ng; }
    }
}

// Q' rows for the chunk + their a16 scales
static void hmxfa_q_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data;
    for (int r = i; r < st->nr; r += n) {
        const float * qr = (const float *) (st->q->data + (size_t) (st->i0 + r) * st->q->nb[1] + (size_t) st->h * st->q->nb[2] + (size_t) st->seq * st->q->nb[3]);
        float * o = st->qs + (size_t) r * st->d;
        for (int c = 0; c < st->d; c += 32) hmxi_stu(o + c, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hmxi_ldu(qr + c), hmxi_ldu(st->sd + c))));
        float am = hmxi_absmax(o, st->d);
        st->sa[r] = am > 0.0f ? am / 32767.0f : 1.0f; st->ia[r] = am > 0.0f ? 32767.0f / am : 0.0f;
    }
}

static void hmxfa_act_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data;
    const int nkt = st->K / 32;
    for (int pp = i; pp < st->nrt * 8; pp += n) {
        int rt = pp / 8, p = pp % 8, ra = rt * 16 + 2 * p, rb = ra + 1;
        hmxi_act_pair(ra < st->nr ? st->act + (size_t) ra * st->act_stride : NULL, rb < st->nr ? st->act + (size_t) rb * st->act_stride : NULL,
                      ra < st->nr ? st->ia[ra] : 0.0f, rb < st->nr ? st->ia[rb] : 0.0f, nkt, st->v_act + (size_t) rt * nkt * 2048, p);
    }
}

static void hmxfa_hmx_fn(void * data) {
    struct hmxfa_state * st = data; struct hmxfa_w * W = st->W;
    for (int rt = 0; rt < st->nrt; rt++) for (int ct = 0; ct < W->nct; ct++) for (int g = 0; g < W->ng; g++) {
        int nb = W->nkt - g * HMXI_G; nb = nb > HMXI_G ? HMXI_G : nb;
        const uint8_t * a  = st->v_act + ((size_t) rt * W->nkt + g * HMXI_G) * 2048;
        const uint8_t * w  = W->tiles + ((size_t) ct * W->nkt + g * HMXI_G) * 1024;
        const uint32_t * b = W->bias + ((size_t) ct * W->ng + g) * 128;
        uint8_t * o = st->v_stage + (((size_t) rt * W->nct + ct) * W->ng + g) * 2048;
        asm volatile("bias = mxmem2(%0)\n" :: "r"((unsigned int) b) : "memory");
        asm volatile("mxclracc\n");
        asm volatile("{\n activation.ub = mxmem(%0, %2):deep\n weight.b = mxmem(%1, %2)\n}\n" :: "r"(a), "r"(w), "r"(nb * 2048 - 1) : "memory");
        asm volatile("mxmem(%0, %1):after:retain.uh = acc:2x1\n" :: "r"(o), "r"(0) : "memory");
        asm volatile("bias = mxmem2(%0)\n" :: "r"((unsigned int) (b + 64)) : "memory");
        asm volatile("mxmem(%0, %1):after.uh = acc:2x1\n" :: "r"(o + 1024), "r"(0) : "memory");
    }
}

static void hmxfa_red_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data; struct hmxfa_w * W = st->W;
    for (int tt = i; tt < st->nrt * W->nct; tt += n) {
        int rt = tt / W->nct, ct = tt % W->nct;
        float * rows[16];
        for (int t = 0; t < 16; t++) {
            int r = rt * 16 + t;
            if (r >= st->nr) { rows[t] = NULL; continue; }
            if (st->out_is_S) rows[t] = st->S + (size_t) r * st->n_kv_pad + ct * 32;
            else rows[t] = (float *) (st->dst->data + (size_t) st->h * st->dst->nb[1] + (size_t) (st->i0 + r) * st->dst->nb[2] + (size_t) st->seq * st->dst->nb[3]) + ct * 32;
        }
        int nv = W->N - ct * 32; nv = nv > 32 ? 32 : nv;
        hmxi_reduce_tile(st->v_stage + (((size_t) rt * W->nct + ct) * W->ng) * 2048, W->ng, rows, st->sa + rt * 16, W->k512 + ct * 32, W->C + ct * 32, 0, nv, NULL);
    }
}

// fp16 bits (64 halfwords in v) -> two f32 vectors in element order (lo = 0..31, hi = 32..63); inf/NaN -> -1e30, exp 0 -> 0
static HVX_VectorPair hmxfa_h2f_vec(HVX_Vector v) {
    const HVX_Vector k7fff = Q6_V_vsplat_R(0x7fff), sgn = Q6_V_vsplat_R(0x8000), k112 = Q6_V_vsplat_R(112 << 23);
    const HVX_Vector k31 = Q6_V_vsplat_R(31), zero = Q6_V_vzero(), neg_big = Q6_V_vsplat_R(0xF149F2CA);   // -1e30f
    HVX_VectorPair X = Q6_Wuw_vzxt_Vuh(v);
    HVX_Vector r[2];
    for (int h = 0; h < 2; h++) {
        HVX_Vector x = h ? Q6_V_hi_W(X) : Q6_V_lo_W(X);
        HVX_Vector mag = Q6_V_vand_VV(x, k7fff), e = Q6_Vw_vasr_VwR(mag, 10);
        HVX_Vector f = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_vand_VV(x, sgn), 16), Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(mag, 13), k112));
        f = Q6_V_vmux_QVV(Q6_Q_vcmp_gt_VwVw(e, zero), f, zero);
        r[h] = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(e, k31), neg_big, f);
    }
    return Q6_W_vshuff_VVR(r[1], r[0], -4);          // interleave even/odd back into element order
}

// softmax rows of S in place -> P; P row scale for the next matmul: max P = 1 -> ia = 32767, sa = 1/(32767 * rowsum)
static void hmxfa_sm_fn(unsigned int n, unsigned int i, void * data) {
    struct hmxfa_state * st = data;
    const HVX_Vector vscale = Q6_V_vsplat_R(hmxi_f2u(st->scale)), vlo = Q6_V_vsplat_R(hmxi_f2u(-80.0f));
    const int nv = st->n_kv / 64 * 64;               // vectorized part (64 per step)
    for (int r = i; r < st->nr; r += n) {
        float * s = st->S + (size_t) r * st->n_kv_pad;
        const int qi = st->i0 + r;
        const uint16_t * mr = st->mask ? (const uint16_t *) (st->mask->data + (size_t) qi * st->mask->nb[1]
                                  + (size_t) (st->h % st->mask->ne[2]) * st->mask->nb[2] + (size_t) (st->seq % st->mask->ne[3]) * st->mask->nb[3]) : NULL;
        HVX_Vector vmax = Q6_V_vsplat_R(0xF149F2CA);
        for (int j = 0; j < nv; j += 64) {
            HVX_Vector x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *) (s + j), vscale));
            HVX_Vector x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(*(HVX_Vector *) (s + j + 32), vscale));
            if (mr) {
                HVX_VectorPair m = hmxfa_h2f_vec(hmxi_ldu(mr + j));
                x0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x0, Q6_V_lo_W(m)));
                x1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x1, Q6_V_hi_W(m)));
            }
            *(HVX_Vector *) (s + j) = x0; *(HVX_Vector *) (s + j + 32) = x1;
            vmax = Q6_Vsf_vmax_VsfVsf(vmax, Q6_Vsf_vmax_VsfVsf(x0, x1));
        }
        float mx = hvx_vec_get_f32(hvx_vec_reduce_max_f32(vmax));
        for (int j = nv; j < st->n_kv; j++) {
            float x = s[j] * st->scale;
            if (mr) { uint16_t mb = mr[j]; x = ((mb & 0x7c00) == 0x7c00) ? -1e30f : x + hmxi_h2f(mb); }
            s[j] = x; mx = x > mx ? x : mx;
        }
        float sum = 0.0f;
        if (mx > -1e29f) {
            const HVX_Vector vm = Q6_V_vsplat_R(hmxi_f2u(mx));
            HVX_Vector vsum = Q6_V_vzero();
            const HVX_Vector log2e = Q6_Vh_vsplat_R(0x3DC5);                // log2(e) in fp16
            for (int j = 0; j < nv; j += 64) {                                  // exp in fp16: exp(x) = exp2(x * log2e)
                HVX_Vector x0 = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(*(HVX_Vector *) (s + j), vm)), vlo);
                HVX_Vector x1 = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(*(HVX_Vector *) (s + j + 32), vm)), vlo);
                HVX_Vector h  = hvx_vec_mul_f16_f16(hvx_vec_f32_to_f16(x0, x1), log2e);
                HVX_VectorPair p = hvx_vec_f16_to_f32(hvx_vec_exp2_f16(h));
                *(HVX_Vector *) (s + j) = Q6_V_lo_W(p); *(HVX_Vector *) (s + j + 32) = Q6_V_hi_W(p);
                vsum = Q6_Vqf32_vadd_Vqf32Vsf(vsum, Q6_V_lo_W(p));
                vsum = Q6_Vqf32_vadd_Vqf32Vsf(vsum, Q6_V_hi_W(p));
            }
            sum = hvx_vec_get_f32(hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(vsum)));
            for (int j = nv; j < st->n_kv; j++) { float x = s[j] - mx; float p = x < -80.0f ? 0.0f : expf(x); s[j] = p; sum += p; }
        } else {
            for (int j = 0; j < st->n_kv; j++) s[j] = 0.0f;
        }
        for (int j = st->n_kv; j < st->n_kv_pad; j++) s[j] = 0.0f;
        st->ia[r] = 32767.0f;
        st->sa[r] = sum > 0.0f ? 1.0f / (32767.0f * sum) : 0.0f;
    }
}

#define HMXFA_T(i, stmt) do { uint64_t _t = HAP_perf_get_time_us(); stmt; st->t[i] += HAP_perf_get_time_us() - _t; } while (0)
static void hmxfa_matmul(struct hmxfa_state * st, const float * act, int act_stride, int K, struct hmxfa_w * W, int out_is_S) {
    struct htp_context * ctx = st->ctx;
    st->act = act; st->act_stride = act_stride; st->K = K; st->W = W; st->out_is_S = out_is_S;
    HMXFA_T(5, worker_pool_run_func(ctx->worker_pool, hmxfa_act_fn, st, st->n_threads));
    HMXFA_T(6, (hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmxfa_hmx_fn, st)), hmx_queue_pop(ctx->hmx_queue)));
    HMXFA_T(7, worker_pool_run_func(ctx->worker_pool, hmxfa_red_fn, st, st->n_threads));
}

static int hmxint_flash_attn(struct htp_ops_context * octx, float scale) {
    const struct htp_tensor * q = octx->src[0], * k = octx->src[1], * v = octx->src[2], * mask = octx->src[3], * dst = octx->dst;
    if (octx->src[4] || q->type != HTP_TYPE_F32 || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16 || dst->type != HTP_TYPE_F32) return HTP_STATUS_NO_SUPPORT;
    if (mask && mask->type != HTP_TYPE_F16) return HTP_STATUS_NO_SUPPORT;
    const int d = q->ne[0], dv = v->ne[0], n_q = q->ne[1], n_head = q->ne[2], n_kv = k->ne[1], n_head_kv = k->ne[2];
    if (d % 32 || dv % 32 || d > HMXFA_MAX_D || dv > HMXFA_MAX_D || n_kv > HMXFA_MAX_KV || n_q < 16 || n_head % n_head_kv) return HTP_STATUS_NO_SUPPORT;

    static struct hmxfa_state st;
    memset(&st, 0, sizeof(st));
    struct htp_context * ctx = octx->ctx;
    st.ctx = ctx; st.q = q; st.k = k; st.v = v; st.mask = mask; st.dst = dst;
    st.d = d; st.dv = dv; st.n_q = n_q; st.n_kv = n_kv; st.n_kv_pad = (n_kv + 31) / 32 * 32;
    st.n_head = n_head; st.n_head_kv = n_head_kv; st.G = n_head / n_head_kv; st.scale = scale;
    st.n_threads = octx->n_threads > 8 ? 8 : octx->n_threads;

    // weights in VTCM
    struct hmxfa_w * W1 = &st.W1, * W2 = &st.W2;
    W1->nct = st.n_kv_pad / 32; W1->nkt = d / 32;           W1->ng = (W1->nkt + HMXI_G - 1) / HMXI_G; W1->N = n_kv; W1->k512 = hmxfa_k512_1; W1->C = hmxfa_C_1;
    W2->nct = dv / 32;          W2->nkt = st.n_kv_pad / 32; W2->ng = (W2->nkt + HMXI_G - 1) / HMXI_G; W2->N = dv;   W2->k512 = hmxfa_k512_2; W2->C = hmxfa_C_2;
    uint8_t * vb = (uint8_t *) ctx->vtcm_base;
    W1->tiles = vb; vb += (size_t) W1->nct * W1->nkt * 1024;
    W2->tiles = vb; vb += (size_t) W2->nct * W2->nkt * 1024;
    W1->bias = (uint32_t *) vb; vb += (size_t) W1->nct * W1->ng * 512;
    W2->bias = (uint32_t *) vb; vb += (size_t) W2->nct * W2->ng * 512;
    vb = (uint8_t *) (((uintptr_t) vb + 4095) & ~(uintptr_t) 4095);   // HMX activation tiles must be 2048-aligned
    const size_t used = (size_t) (vb - (uint8_t *) ctx->vtcm_base);
    // per row tile: act max(nkt) * 2048 + stage max(nct * ng) * 2048
    const int nkt_max = W1->nkt > W2->nkt ? W1->nkt : W2->nkt;
    const int ncg_max = (W1->nct * W1->ng) > (W2->nct * W2->ng) ? W1->nct * W1->ng : W2->nct * W2->ng;
    // per row tile: act + stage + S rows + Q' rows, all in VTCM
    const size_t per_rt = (size_t) nkt_max * 2048 + (size_t) ncg_max * 2048 + (size_t) 16 * st.n_kv_pad * 4 + (size_t) 16 * d * 4;
    if (used + per_rt > ctx->vtcm_size) return HTP_STATUS_NO_SUPPORT;
    int rc = (int) ((ctx->vtcm_size - used) / per_rt); rc = rc > 64 ? 64 : rc;   // row tiles per chunk (<= 1024 rows)
    st.v_act = vb; vb += (size_t) rc * nkt_max * 2048;
    st.v_stage = vb; vb += (size_t) rc * ncg_max * 2048;
    float * vS = (float *) vb; vb += (size_t) rc * 16 * st.n_kv_pad * 4;
    float * vQ = (float *) vb; vb += (size_t) rc * 16 * d * 4;

    const int rows_max = rc * 16;
    size_t need_q = 1, need_s = 1;
    if (hmxfa_qs_sz < need_q) { free(hmxfa_qs_buf); hmxfa_qs_buf = memalign(128, need_q * 4); hmxfa_qs_sz = hmxfa_qs_buf ? need_q : 0; }
    if (hmxfa_S_sz < need_s)  { free(hmxfa_S_buf);  hmxfa_S_buf  = memalign(128, need_s * 4); hmxfa_S_sz  = hmxfa_S_buf ? need_s : 0; }
    size_t need_kv = (size_t) st.n_kv_pad * d + (size_t) st.n_kv_pad * 4 + (size_t) st.n_kv_pad * 2 + 256 + (size_t) st.n_kv_pad * dv * 4 + 256;
    if (hmxfa_kv_sz < need_kv) { free(hmxfa_kv_buf); hmxfa_kv_buf = memalign(128, need_kv); hmxfa_kv_sz = hmxfa_kv_buf ? need_kv : 0; }
    if (!hmxfa_qs_buf || !hmxfa_S_buf || !hmxfa_kv_buf) return HTP_STATUS_NO_SUPPORT;
    {
        uint8_t * p = hmxfa_kv_buf;
        st.vf  = (float *) p;    p += (size_t) st.n_kv_pad * dv * 4;
        st.k8  = (int8_t *) p;   p += (size_t) st.n_kv_pad * d;
        p = (uint8_t *) (((uintptr_t) p + 127) & ~(uintptr_t) 127);
        st.kS  = (int32_t *) p;  p += (size_t) st.n_kv_pad * 4;
        st.kS0 = (uint16_t *) p;
    }
    st.qs = vQ; st.S = vS;

    for (int seq = 0; seq < (int) q->ne[3]; seq++) {
        st.seq = seq;
        for (int hk = 0; hk < n_head_kv; hk++) {
            st.hk = hk;
            { struct hmxfa_state * stp = &st; struct hmxfa_state * st = stp; HMXFA_T(0, worker_pool_run_func(ctx->worker_pool, hmxfa_sd_fn, st, st->n_threads)); }
            for (int c = 0; c < d; c++) {
                float m = 0.0f; for (int t = 0; t < st.n_threads; t++) m = st.spart[t][c] > m ? st.spart[t][c] : m;
                st.sd[c] = m > 0.0f ? m : 1.0f; st.isd[c] = 1.0f / st.sd[c];
            }
            { struct hmxfa_state * stp = &st; struct hmxfa_state * st = stp; HMXFA_T(1, worker_pool_run_func(ctx->worker_pool, hmxfa_rows_fn, st, st->n_threads)); }
            for (int c = 0; c < dv; c++) {
                float m = 0.0f; for (int t = 0; t < st.n_threads; t++) m = st.vmax_part[t][c] > m ? st.vmax_part[t][c] : m;
                st.sVb[c] = m > 0.0f ? hmxi_f16_round_up_bits(m / 127.0f) : 0;
                st.sV[c] = st.sVb[c] ? hmxi_h2f(st.sVb[c]) : 0.0f; st.iV[c] = st.sV[c] > 0.0f ? 1.0f / st.sV[c] : 0.0f;
            }
            { struct hmxfa_state * stp = &st; struct hmxfa_state * st = stp;
              HMXFA_T(2, worker_pool_run_func(ctx->worker_pool, hmxfa_w1_fn, st, st->n_threads));
              HMXFA_T(8, worker_pool_run_func(ctx->worker_pool, hmxfa_w2_fn, st, st->n_threads)); }
            for (int g = 0; g < st.G; g++) {
                st.h = hk * st.G + g;
                for (int i0 = 0; i0 < n_q; i0 += rows_max) {
                    st.i0 = i0; st.nr = n_q - i0 < rows_max ? n_q - i0 : rows_max; st.nrt = (st.nr + 15) / 16;
                    for (int r = st.nr; r < st.nrt * 16; r++) { st.sa[r] = 0.0f; st.ia[r] = 0.0f; }
                    { struct hmxfa_state * stp = &st; struct hmxfa_state * st = stp; HMXFA_T(3, worker_pool_run_func(ctx->worker_pool, hmxfa_q_fn, st, st->n_threads)); }
                    hmxfa_matmul(&st, st.qs, d, d, W1, 1);
                    { struct hmxfa_state * stp = &st; struct hmxfa_state * st = stp; HMXFA_T(4, worker_pool_run_func(ctx->worker_pool, hmxfa_sm_fn, st, st->n_threads)); }
                    hmxfa_matmul(&st, st.S, st.n_kv_pad, st.n_kv_pad, W2, 0);
                }
            }
        }
    }
    static int nlog = 0;
    if (nlog++ < 0) {
        FARF(ALWAYS, "hmxfa: d %d nq %d nkv %d", d, n_q, n_kv);
        FARF(ALWAYS, "hmxfa:   heads %d kvh %d rc %d", n_head, n_head_kv, rc);
        FARF(ALWAYS, "hmxfa:   sd %u rows %u w1 %u w2 %u", (unsigned) st.t[0], (unsigned) st.t[1], (unsigned) st.t[2], (unsigned) st.t[8]);
        FARF(ALWAYS, "hmxfa:   q %u sm %u act %u", (unsigned) st.t[3], (unsigned) st.t[4], (unsigned) st.t[5]);
        FARF(ALWAYS, "hmxfa:   hmx %u red %u", (unsigned) st.t[6], (unsigned) st.t[7]);
    }
    return HTP_STATUS_OK;
}
