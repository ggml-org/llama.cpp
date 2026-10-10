#include "common.h"
#include "dequantize.h"

constant short FC_mul_mv_mma_nsg  [[function_constant(FC_MUL_MV_MMA + 0)]];
constant short FC_mul_mv_mma_ne12 [[function_constant(FC_MUL_MV_MMA + 1)]];
constant short FC_mul_mv_mma_r2   [[function_constant(FC_MUL_MV_MMA + 2)]];
constant short FC_mul_mv_mma_r3   [[function_constant(FC_MUL_MV_MMA + 3)]];
constant int   FC_mul_mv_mma_ne00 [[function_constant(FC_MUL_MV_MMA + 4)]];
constant bool  FC_mul_mv_mma_add  [[function_constant(FC_MUL_MV_MMA + 5)]];

// a lane of a few-row MMA tile: A fragment row fm, B fragment columns fn and fn + 1
struct mul_mv_mma_tile {
    short    fm;
    short    fn;
    int      i01;
    int      i11;
    int      i1m;
    uint64_t offset0;
    uint64_t offset1;
};

// the A fragment row and the first B fragment column that lane l holds in an 8x8 simdgroup matrix
inline short mul_mv_mma_lane_fm(ushort l) { return ((l/4) & 4) + ((l/2) % 4); }
inline short mul_mv_mma_lane_fn(ushort l) { return ((l/4) & 2)*2 + (l % 2)*2; }

template<short NT, short RT>
inline mul_mv_mma_tile mul_mv_mma_tile_init(constant ggml_metal_kargs_mul_mv_ext & args, uint3 tgpig, ushort tiisg) {
    mul_mv_mma_tile tile;
    tile.fm  = mul_mv_mma_lane_fm(tiisg);
    tile.fn  = mul_mv_mma_lane_fn(tiisg);
    tile.i01 = tgpig.x*(8*NT);
    tile.i11 = tgpig.y*(8*RT);
    tile.i1m = tgpig.z;

    const int i12 = tile.i1m%FC_mul_mv_mma_ne12;
    const int i13 = tile.i1m/FC_mul_mv_mma_ne12;

    tile.offset0 = (i12/FC_mul_mv_mma_r2)*args.nb02 + (i13/FC_mul_mv_mma_r3)*args.nb03;
    tile.offset1 = i12*args.nb12 + i13*args.nb13;

    return tile;
}

// a lane of a few-row MMA tile of MUL_MAT_ID: the src1 rows are the tokens that kernel_mul_mm_id_map0 listed for an expert
struct mul_mv_mma_tile_id {
    short    fm;
    short    fn;
    int      i01;
    int      i11; // first entry of the token list
    int      n1;  // tokens routed to the expert
    uint64_t offset0;
    device const int32_t * ids; // token*ne20 + slot per entry
};

// tgpig.z indexes the experts with tokens; a tile past them gets no tokens. The kernel loops over the 8*RT-row tiles of its expert
template<short NT, short RT>
inline mul_mv_mma_tile_id mul_mv_mma_tile_id_init(
        constant ggml_metal_kargs_mul_mv_mma_id & args, device const char * htpe, device const char * hids, device const char * hact, uint3 tgpig, ushort tiisg) {
    device const int32_t * act = (device const int32_t *) hact;

    const bool active = (int) tgpig.z < act[0];
    const int  im     = active ? act[1 + tgpig.z] : 0;

    mul_mv_mma_tile_id tile;
    tile.fm      = mul_mv_mma_lane_fm(tiisg);
    tile.fn      = mul_mv_mma_lane_fn(tiisg);
    tile.i01     = tgpig.x*(8*NT);
    tile.i11     = 0;
    tile.n1      = active ? ((device const uint32_t *) htpe)[im] : 0;
    tile.offset0 = (uint64_t) im*args.nb02;
    tile.ids     = (device const int32_t *) hids + im*args.ne21;

    return tile;
}

// the src0 row of A fragment row fm in 8-row tile t, clamped to the last row
template<typename T, typename A>
inline device const char * mul_mv_mma_src0_row(thread const T & tile, constant A & args, device const char * src0, short t) {
    const int r = min(tile.i01 + 8*t + tile.fm, args.ne01 - 1);
    return src0 + tile.offset0 + (uint64_t) r*args.nb01;
}

// the src1 row of B fragment column fn + e in 8-row tile rt, clamped to the last row
inline device const float * mul_mv_mma_src1_row(
        thread const mul_mv_mma_tile & tile, constant ggml_metal_kargs_mul_mv_ext & args, device const char * src1, short rt, short e) {
    const int r = min(tile.i11 + 8*rt + tile.fn + e, args.ne11 - 1);
    return (device const float *) (src1 + tile.offset1 + (uint64_t) r*args.nb11);
}

inline device const float * mul_mv_mma_src1_row(
        thread const mul_mv_mma_tile_id & tile, constant ggml_metal_kargs_mul_mv_mma_id & args, device const char * src1, short rt, short e) {
    const int id = tile.ids[min(tile.i11 + 8*rt + tile.fn + e, tile.n1 - 1)];
    return (device const float *) (src1 + (uint64_t) ((id % args.ne20) % args.ne11)*args.nb11 + (uint64_t) (id / args.ne20)*args.nb12);
}

constexpr constant static ushort mma_f16_1024_bits = 0x6400;
constexpr constant static half   mma_f16_1024      = 1024.0h;

// the halves 1024 + q for integers q < 1024: exact normal values, unlike the subnormal q*2^-24 that Metal may flush to zero
inline half2 mul_mv_mma_1024_plus(ushort2 q) {
    return as_type<half2>(q | mma_f16_1024_bits);
}

// adds up the K slices of the NSG simdgroups for an 8*NT x 8*RT output tile and writes it, plus the residual in src2 if FC_mul_mv_mma_add
template<short NT, short RT>
inline void mul_mv_mma_store(
        thread float (&acc)[RT][NT][2],
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src2,
        device char * dst,
        threadgroup char * shmem,
        thread const mul_mv_mma_tile & tile, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_mma_nsg;

    threadgroup float * red = (threadgroup float *) shmem;

    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 0] = acc[rt][t][0];
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 1] = acc[rt][t][1];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device       float * dst_f32 = (device       float *) dst  + (uint64_t) tile.i1m*args.ne0*args.ne1;
    device const float * res_f32 = (device const float *) src2 + (uint64_t) tile.i1m*args.ne0*args.ne1;

    for (short idx = sgitg*32 + tiisg; idx < RT*NT*64; idx += NSG*32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg*(RT*NT*64) + idx];
        }

        const short rt = idx/(NT*64);
        const short t  = (idx/64) % NT;
        const short l  = (idx % 64)/2;
        const short e  = idx % 2;

        const int r0 = tile.i01 + 8*t  + mul_mv_mma_lane_fm(l);
        const int r1 = tile.i11 + 8*rt + mul_mv_mma_lane_fn(l) + e;

        if (r0 < args.ne01 && r1 < args.ne11) {
            const uint64_t i = (uint64_t) r1*args.ne0 + r0;
            dst_f32[i] = FC_mul_mv_mma_add ? sum + res_f32[i] : sum;
        }
    }
}

// adds up the K slices like the mul_mv_mma_store above and writes each token row of the expert tile to the dst row of its token and slot
template<short NT, short RT>
inline void mul_mv_mma_store(
        thread float (&acc)[RT][NT][2],
        constant ggml_metal_kargs_mul_mv_mma_id & args,
        device const char * src2,
        device char * dst,
        threadgroup char * shmem,
        thread const mul_mv_mma_tile_id & tile, ushort tiisg, ushort sgitg) {
    const short NSG = FC_mul_mv_mma_nsg;

    threadgroup float * red = (threadgroup float *) shmem;

    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 0] = acc[rt][t][0];
            red[((sgitg*RT + rt)*NT + t)*64 + 2*tiisg + 1] = acc[rt][t][1];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short idx = sgitg*32 + tiisg; idx < RT*NT*64; idx += NSG*32) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg*(RT*NT*64) + idx];
        }

        const short rt = idx/(NT*64);
        const short t  = (idx/64) % NT;
        const short l  = (idx % 64)/2;
        const short e  = idx % 2;

        const int r0 = tile.i01 + 8*t  + mul_mv_mma_lane_fm(l);
        const int r1 = tile.i11 + 8*rt + mul_mv_mma_lane_fn(l) + e;

        if (r0 < args.ne01 && r1 < tile.n1) {
            const int id = tile.ids[r1];
            ((device float *) dst)[((uint64_t) (id / args.ne20)*args.ne1 + id % args.ne20)*args.ne0 + r0] = sum;
        }
    }
}

// the per-type parts of kernel_mul_mv_mma_blk for 32-weight blocks. a src1 block splits into halves b0 and b1,
// and MMA step s uses half b1 when b1_step(s) and the .y value of a pair when y_step(s)
// q4_0: A lane (m, j) holds qs ushorts j and j + 1 (j even); a high nibble stays in place as 16*q, so b1 is divided by 16.
// B lane k = fm holds src1 values 2*k, 2*k + 1 (b0, low nibbles) and 2*k + 16, 2*k + 17 (b1, high nibbles) of a block
struct mul_mv_mma_q4_0 {
    typedef block_q4_0 block;
    typedef ushort2    quants;

    // weights per block, and the float2 offset of b1 in a src1 block
    enum { qk = QK4_0, b1 = QK4_0/4 };

    static short a_off(short fn) { return 1 + fn; }
    static short b_off(short fm) { return 2*fm; }

    static quants load(device const ushort * qs) { return ushort2(qs[0], qs[1]); }
    static quants prep(quants q) { return q; }
    static float2 prep_b1(float2 v) {
        constexpr float hi_scale = 1.0f/16;
        return v*hi_scale;
    }

    static bool b1_step(short s) { return s % 2 != 0; }
    static bool y_step (short s) { return s >= 2; }

    static half2 frag(quants q, short s) {
        constexpr ushort lo_mask = 0x000F;
        constexpr ushort hi_mask = 0x00F0;
        constexpr half   lo_zero = 8.0h;
        constexpr half   hi_zero = 16*lo_zero;

        const ushort2 qq = s < 2 ? q : q >> 8;
        return s % 2 == 0 ? mul_mv_mma_1024_plus(qq & lo_mask) - (mma_f16_1024 + lo_zero) : mul_mv_mma_1024_plus(qq & hi_mask) - (mma_f16_1024 + hi_zero);
    }
};

// q8_0: A lane (m, j) holds qs bytes 4*j .. 4*j + 7 (j even); flipping the sign bit of a quant byte gives the unsigned q + 128.
// B lane k = fm holds src1 values b, b + 1 (b0) and b + 4, b + 5 (b1) of a block, b = 8*(k/2) + 2*(k%2)
struct mul_mv_mma_q8_0 {
    typedef block_q8_0 block;
    typedef ushort4    quants;

    // weights per block, and the float2 offset of b1 in a src1 block
    enum { qk = QK8_0, b1 = 2 };

    static short a_off(short fn) { return 1 + 2*fn; }
    static short b_off(short fm) { return 8*(fm/2) + 2*(fm%2); }

    static quants load(device const ushort * qs) { return ushort4(qs[0], qs[1], qs[2], qs[3]); }
    static quants prep(quants q) {
        constexpr ushort sign_bits = 0x8080;
        return q ^ ushort4(sign_bits);
    }
    static float2 prep_b1(float2 v) { return v; }

    static bool b1_step(short s) { return s >= 2; }
    static bool y_step (short s) { return s % 2 != 0; }

    static half2 frag(quants q, short s) {
        constexpr ushort byte_mask = 0x00FF;
        constexpr half   q_bias    = mma_f16_1024 + 128.0h;

        const ushort2 w  = s < 2 ? q.xy : q.zw;
        const ushort2 qq = s % 2 == 0 ? w : w >> 8;
        return mul_mv_mma_1024_plus(qq & byte_mask) - q_bias;
    }
};

template<typename Q, short NT>
inline void load_mma_blk_a(device const ushort * const x[NT], int off, short fn, thread typename Q::quants * q, thread float * d) {
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const ushort * qs = x[t] + off;
        q[t] = Q::load(qs);
        d[t] = as_type<half>(*(qs - Q::a_off(fn)));
    }
}

template<typename Q, short RT>
inline void load_mma_blk_b(device const float2 * const y[RT][2], int ib, thread float2 (*b0)[2], thread float2 (*b1)[2]) {
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            b0[rt][e] = y[rt][e][ib*(Q::qk/2)];
            b1[rt][e] = y[rt][e][ib*(Q::qk/2) + Q::b1];
        }
    }
}

// few-row mat-mat (2..16 src1 rows) on 8x8 simdgroup matrices for 32-weight block types: a threadgroup reads each weight once
// for 8*NT src0 rows x 8*RT src1 rows, and its NSG simdgroups split K
template<short NT, short RT, typename Q, typename T, typename A>
__attribute__((always_inline)) inline void mul_mv_mma_blk_impl(
        constant A & args,
        thread const T & tile,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_mma_nsg;

    device const ushort * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const ushort *) mul_mv_mma_src0_row(tile, args, src0, t) + Q::a_off(tile.fn);
    }

    device const float2 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float2 *) (mul_mv_mma_src1_row(tile, args, src1, rt, e) + Q::b_off(tile.fm));
        }
    }

    float acc[RT][NT][2] = {};

    const int nb = FC_mul_mv_mma_ne00/Q::qk;

    // a block is d, then the quants
    constexpr short us_blk = sizeof(typename Q::block)/2;

    typename Q::quants q[NT];
    float  d[NT];
    float2 b0[RT][2];
    float2 b1[RT][2];

    const int ib0 = min((int) sgitg, nb - 1);
    load_mma_blk_a<Q, NT>(x, ib0*us_blk, tile.fn, q, d);
    load_mma_blk_b<Q, RT>(y, ib0, b0, b1);

    for (int ib = sgitg; ib < nb; ib += NSG) {
        typename Q::quants qc[NT];
        float  dc[NT];
        float2 b0c[RT][2];
        float2 b1c[RT][2];

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            qc[t] = Q::prep(q[t]);
            dc[t] = d[t];
        }
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                b0c[rt][e] = b0[rt][e];
                b1c[rt][e] = Q::prep_b1(b1[rt][e]);
            }
        }

        const int ibn = min(ib + NSG, nb - 1);
        load_mma_blk_a<Q, NT>(x, ibn*us_blk, tile.fn, q, d);
        load_mma_blk_b<Q, RT>(y, ibn, b0, b1);

        simdgroup_float8x8 mp[RT][NT];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short t = 0; t < NT; ++t) {
                mp[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }
        }

        FOR_UNROLL (short s = 0; s < 4; ++s) {
            simdgroup_float8x8 mb[RT];
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                const float2 v0 = Q::b1_step(s) ? b1c[rt][0] : b0c[rt][0];
                const float2 v1 = Q::b1_step(s) ? b1c[rt][1] : b0c[rt][1];
                mb[rt].thread_elements()[0] = Q::y_step(s) ? v0.y : v0.x;
                mb[rt].thread_elements()[1] = Q::y_step(s) ? v1.y : v1.x;
            }

            FOR_UNROLL (short t = 0; t < NT; ++t) {
                const half2 h = Q::frag(qc[t], s);

                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = h.x;
                ma.thread_elements()[1] = h.y;

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mp[rt][t], ma, mb[rt], mp[rt][t]);
                }
            }
        }

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                acc[rt][t][0] = fma(dc[t], mp[rt][t].thread_elements()[0], acc[rt][t][0]);
                acc[rt][t][1] = fma(dc[t], mp[rt][t].thread_elements()[1], acc[rt][t][1]);
            }
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, src2, dst, shmem, tile, tiisg, sgitg);
}

template<short NT, short RT, typename Q>
kernel void kernel_mul_mv_mma_blk(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(args, tgpig, tiisg);

    mul_mv_mma_blk_impl<NT, RT, Q>(args, tile, src0, src1, dst, src2, shmem, tiisg, sgitg);
}

template<short NT, short RT, typename Q>
kernel void kernel_mul_mv_mma_id_blk(
        constant ggml_metal_kargs_mul_mv_mma_id & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * htpe,
        device const char * hids,
        device const char * hact,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_mv_mma_tile_id tile = mul_mv_mma_tile_id_init<NT, RT>(args, htpe, hids, hact, tgpig, tiisg);
    for (; tile.i11 < tile.n1; tile.i11 += 8*RT) {
        mul_mv_mma_blk_impl<NT, RT, Q>(args, tile, src0, src1, dst, nullptr, shmem, tiisg, sgitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

typedef decltype(kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q4_0>) mul_mv_mma_t;
typedef decltype(kernel_mul_mv_mma_id_blk<4, 1, mul_mv_mma_q4_0>) mul_mv_mma_id_t; // MUL_MAT_ID runs only nt4 rt1 (ggml_metal_op_mul_mat_mma_tiling)

template [[host_name("kernel_mul_mv_mma_q4_0_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<1, 1, mul_mv_mma_q4_0>;
template [[host_name("kernel_mul_mv_mma_q4_0_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<2, 1, mul_mv_mma_q4_0>;
template [[host_name("kernel_mul_mv_mma_q4_0_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q4_0>;
template [[host_name("kernel_mul_mv_mma_q4_0_f32_nt1_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<1, 2, mul_mv_mma_q4_0>;
template [[host_name("kernel_mul_mv_mma_q4_0_f32_nt2_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<2, 2, mul_mv_mma_q4_0>;
template [[host_name("kernel_mul_mv_mma_q4_0_f32_nt4_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<4, 2, mul_mv_mma_q4_0>;
template [[host_name("kernel_mul_mv_mma_id_q4_0_f32_nt4_rt1")]] kernel mul_mv_mma_id_t kernel_mul_mv_mma_id_blk<4, 1, mul_mv_mma_q4_0>;

template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<1, 1, mul_mv_mma_q8_0>;
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<2, 1, mul_mv_mma_q8_0>;
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_blk<4, 1, mul_mv_mma_q8_0>;
template [[host_name("kernel_mul_mv_mma_id_q8_0_f32_nt4_rt1")]] kernel mul_mv_mma_id_t kernel_mul_mv_mma_id_blk<4, 1, mul_mv_mma_q8_0>;

// q5_K scale and min of sub-block j from the 12 packed bytes, held as 3 words
inline float2 mul_mv_mma_q5_K_scale_min(thread const uint * w, short j) {
    if (j < 4) {
        return float2((w[0] >> 8*j) & 63, (w[1] >> 8*j) & 63);
    }
    const short k = 8*(j - 4);
    return float2(((w[2] >> k) & 0xF) | (((w[0] >> (k + 6)) & 3) << 4), ((w[2] >> (k + 4)) & 0xF) | (((w[1] >> (k + 6)) & 3) << 4));
}

// the A fragments of one qs/qh word: lo holds the 5-bit quants q of the low-nibble sub-block of a pair, hi holds 16*q of the high-nibble sub-block.
// step e0 takes bytes 0 and 2 of the word, step e1 takes bytes 1 and 3; hs holds the qh bits of the pair.
inline void mul_mv_mma_q5_K_frags(uint q, uint hs, thread half2 * lo, thread half2 * hi) {
    const ushort2 qw = as_type<ushort2>(q);
    const ushort2 hw = as_type<ushort2>(hs);

    lo[0] = mul_mv_mma_1024_plus((qw        & 0x000F) | ((hw << 4) & 0x0010)) - mma_f16_1024;
    lo[1] = mul_mv_mma_1024_plus(((qw >> 8) & 0x000F) | ((hw >> 4) & 0x0010)) - mma_f16_1024;
    hi[0] = mul_mv_mma_1024_plus((qw        & 0x00F0) | ((hw << 7) & 0x0100)) - mma_f16_1024;
    hi[1] = mul_mv_mma_1024_plus(((qw >> 8) & 0x00F0) | ((hw >> 1) & 0x0100)) - mma_f16_1024;
}

struct mul_mv_mma_q5_K_a {
    uint2 q;
    uint2 h;
    uint  sc[3];
    uint  dm;
};

template<short NT>
inline void load_q5_K_mma_a(device const block_q5_K * const x[NT], int ip, short fn, thread mul_mv_mma_q5_K_a * a) {
    constexpr short pairs = QK_K/64;

    FOR_UNROLL (short t = 0; t < NT; ++t) {
        device const block_q5_K * xb = x[t] + ip/pairs;
        device const uint * sp = (device const uint *) xb->scales;

        a[t].q     = *((device const uint2 *) (xb->qs + 32*(ip%pairs)) + fn/2);
        a[t].h     = *((device const uint2 *) xb->qh + fn/2);
        a[t].sc[0] = sp[0];
        a[t].sc[1] = sp[1];
        a[t].sc[2] = sp[2];
        a[t].dm    = *((device const uint *) xb);
    }
}

// xs[rt][e][h]: src1 values of sub-block h of the pair, for the steps (b, b + 1) and (b + 4, b + 5)
template<short RT>
inline void load_q5_K_mma_b(device const float2 * const y[RT][2], int ip, thread float2 (*xs)[2][2][2]) {
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            FOR_UNROLL (short h = 0; h < 2; ++h) {
                xs[rt][e][h][0] = y[rt][e][ip*32 + h*16 + 0];
                xs[rt][e][h][1] = y[rt][e][ip*32 + h*16 + 2];
            }
        }
    }
}

// few-row mat-mat for q5_K over pairs of 32-weight sub-blocks, laid out like mul_mv_mma_q4_0: A lane (m, j) holds qs and qh bytes 4*j .. 4*j + 7 of a pair (j even).
// the 1/16 of the in-place high nibbles and the sub-block scales go into the accumulation; the mins are removed with the src1 sums.
template<short NT, short RT, typename T, typename A>
__attribute__((always_inline)) inline void mul_mv_mma_q5_K_impl(
        constant A & args,
        thread const T & tile,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_mma_nsg;

    constexpr float hi_scale = 1.0f/16;
    constexpr short pairs    = QK_K/64;

    device const block_q5_K * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q5_K *) mul_mv_mma_src0_row(tile, args, src0, t);
    }

    // B lane k = fm holds src1 values b, b + 1, b + 4, b + 5 of each sub-block, b = 8*(k/2) + 2*(k%2)
    device const float2 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float2 *) (mul_mv_mma_src1_row(tile, args, src1, rt, e) + 8*(tile.fm/2) + 2*(tile.fm%2));
        }
    }

    float acc[RT][NT][2] = {};

    const int np = FC_mul_mv_mma_ne00/64;

    mul_mv_mma_q5_K_a an[NT];
    float2 xn[RT][2][2][2];

    const int ip0 = min((int) sgitg, np - 1);
    load_q5_K_mma_a<NT>(x, ip0, tile.fn, an);
    load_q5_K_mma_b<RT>(y, ip0, xn);

    for (int ip = sgitg; ip < np; ip += NSG) {
        const short p = ip%pairs;

        mul_mv_mma_q5_K_a ac[NT];
        float2 xs[RT][2][2][2];
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            ac[t] = an[t];
        }
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    xs[rt][e][h][0] = xn[rt][e][h][0];
                    xs[rt][e][h][1] = xn[rt][e][h][1];
                }
            }
        }

        const int ipn = min(ip + NSG, np - 1);
        load_q5_K_mma_a<NT>(x, ipn, tile.fn, an);
        load_q5_K_mma_b<RT>(y, ipn, xn);

        float c[RT][2][2];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short e = 0; e < 2; ++e) {
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    float u = xs[rt][e][h][0].x + xs[rt][e][h][0].y + xs[rt][e][h][1].x + xs[rt][e][h][1].y;
                    u += simd_shuffle_xor(u, 2);
                    u += simd_shuffle_xor(u, 4);
                    u += simd_shuffle_xor(u, 16);
                    c[rt][e][h] = u;
                }
            }
        }

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            const float2 dm  = float2(as_type<half2>(ac[t].dm));
            const float2 sm0 = mul_mv_mma_q5_K_scale_min(ac[t].sc, 2*p + 0);
            const float2 sm1 = mul_mv_mma_q5_K_scale_min(ac[t].sc, 2*p + 1);

            half2 a[2][2][2];
            mul_mv_mma_q5_K_frags(ac[t].q.x, ac[t].h.x >> 2*p, a[0][0], a[1][0]);
            mul_mv_mma_q5_K_frags(ac[t].q.y, ac[t].h.y >> 2*p, a[0][1], a[1][1]);

            FOR_UNROLL (short hh = 0; hh < 2; ++hh) {
                simdgroup_float8x8 mp[RT];
                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    mp[rt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                }

                FOR_UNROLL (short s = 0; s < 4; ++s) {
                    simdgroup_half8x8 ma;
                    ma.thread_elements()[0] = a[hh][s/2][s%2].x;
                    ma.thread_elements()[1] = a[hh][s/2][s%2].y;

                    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                        simdgroup_float8x8 mb;
                        mb.thread_elements()[0] = s % 2 == 0 ? xs[rt][0][hh][s/2].x : xs[rt][0][hh][s/2].y;
                        mb.thread_elements()[1] = s % 2 == 0 ? xs[rt][1][hh][s/2].x : xs[rt][1][hh][s/2].y;

                        simdgroup_multiply_accumulate(mp[rt], ma, mb, mp[rt]);
                    }
                }

                const float2 smh = hh == 0 ? sm0 : sm1;
                const float  dsc = dm.x*smh.x*(hh == 0 ? 1.0f : hi_scale);
                const float  dmn = dm.y*smh.y;

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    acc[rt][t][0] = fma(dsc, mp[rt].thread_elements()[0], fma(-dmn, c[rt][0][hh], acc[rt][t][0]));
                    acc[rt][t][1] = fma(dsc, mp[rt].thread_elements()[1], fma(-dmn, c[rt][1][hh], acc[rt][t][1]));
                }
            }
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, src2, dst, shmem, tile, tiisg, sgitg);
}

template<short NT, short RT>
kernel void kernel_mul_mv_mma_q5_K_f32(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(args, tgpig, tiisg);

    mul_mv_mma_q5_K_impl<NT, RT>(args, tile, src0, src1, dst, src2, shmem, tiisg, sgitg);
}

template<short NT, short RT>
kernel void kernel_mul_mv_mma_id_q5_K_f32(
        constant ggml_metal_kargs_mul_mv_mma_id & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * htpe,
        device const char * hids,
        device const char * hact,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_mv_mma_tile_id tile = mul_mv_mma_tile_id_init<NT, RT>(args, htpe, hids, hact, tgpig, tiisg);
    for (; tile.i11 < tile.n1; tile.i11 += 8*RT) {
        mul_mv_mma_q5_K_impl<NT, RT>(args, tile, src0, src1, dst, nullptr, shmem, tiisg, sgitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("kernel_mul_mv_mma_q5_K_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q5_K_f32<1, 1>;
template [[host_name("kernel_mul_mv_mma_q5_K_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q5_K_f32<2, 1>;
template [[host_name("kernel_mul_mv_mma_q5_K_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q5_K_f32<4, 1>;
template [[host_name("kernel_mul_mv_mma_q5_K_f32_nt1_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q5_K_f32<1, 2>;
template [[host_name("kernel_mul_mv_mma_q5_K_f32_nt2_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q5_K_f32<2, 2>;
template [[host_name("kernel_mul_mv_mma_q5_K_f32_nt4_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_q5_K_f32<4, 2>;
template [[host_name("kernel_mul_mv_mma_id_q5_K_f32_nt4_rt1")]] kernel mul_mv_mma_id_t kernel_mul_mv_mma_id_q5_K_f32<4, 1>;

// few-row mat-mat for any type with a 16-weight dequantizer: a lane dequantizes 16 consecutive weights of a 64-weight chunk once for all src1 rows.
// MMA step s at MMA-k index j reads chunk weight 16*(j/2) + 8*(j%2) + s.
template<short NT, short RT, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &), typename T, typename A>
__attribute__((always_inline)) inline void mul_mv_mma_gen_impl(
        constant A & args,
        thread const T & tile,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_mma_nsg;

    device const block_q * x[NT];
    FOR_UNROLL (short t = 0; t < NT; ++t) {
        x[t] = (device const block_q *) mul_mv_mma_src0_row(tile, args, src0, t);
    }

    // B lane j = fm reads chunk values 16*(fm/2) + 8*(fm%2) .. +7
    device const float4 * y[RT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short e = 0; e < 2; ++e) {
            y[rt][e] = (device const float4 *) mul_mv_mma_src1_row(tile, args, src1, rt, e) + 4*(tile.fm/2) + 2*(tile.fm%2);
        }
    }

    simdgroup_float8x8 mc[RT][NT];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            mc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    const int nch = args.ne00/64;

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[RT][8];
        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            const float4 a0 = y[rt][0][16*g + 0];
            const float4 a1 = y[rt][0][16*g + 1];
            const float4 b0 = y[rt][1][16*g + 0];
            const float4 b1 = y[rt][1][16*g + 1];
            FOR_UNROLL (short s = 0; s < 4; ++s) {
                mb[rt][s    ].thread_elements()[0] = a0[s];
                mb[rt][s    ].thread_elements()[1] = b0[s];
                mb[rt][s + 4].thread_elements()[0] = a1[s];
                mb[rt][s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4*g + tile.fn/2;

        FOR_UNROLL (short t = 0; t < NT; ++t) {
            float4x4 w;
            dequantize_func(x[t] + ci/nl, ci%nl, w);

            FOR_UNROLL (short s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s/4    ][s%4];
                ma.thread_elements()[1] = w[s/4 + 2][s%4];

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mc[rt][t], ma, mb[rt][s], mc[rt][t]);
                }
            }
        }
    }

    float acc[RT][NT][2];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short t = 0; t < NT; ++t) {
            acc[rt][t][0] = mc[rt][t].thread_elements()[0];
            acc[rt][t][1] = mc[rt][t].thread_elements()[1];
        }
    }

    mul_mv_mma_store<NT, RT>(acc, args, src2, dst, shmem, tile, tiisg, sgitg);
}

template<short NT, short RT, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_mul_mv_mma_gen(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * src2,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const mul_mv_mma_tile tile = mul_mv_mma_tile_init<NT, RT>(args, tgpig, tiisg);

    mul_mv_mma_gen_impl<NT, RT, block_q, nl, dequantize_func>(args, tile, src0, src1, dst, src2, shmem, tiisg, sgitg);
}

template<short NT, short RT, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_mul_mv_mma_id_gen(
        constant ggml_metal_kargs_mul_mv_mma_id & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        device const char * htpe,
        device const char * hids,
        device const char * hact,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_mv_mma_tile_id tile = mul_mv_mma_tile_id_init<NT, RT>(args, htpe, hids, hact, tgpig, tiisg);
    for (; tile.i11 < tile.n1; tile.i11 += 8*RT) {
        mul_mv_mma_gen_impl<NT, RT, block_q, nl, dequantize_func>(args, tile, src0, src1, dst, nullptr, shmem, tiisg, sgitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define MUL_MV_MMA_GEN(tname, bq, nl, deq) \
template [[host_name("kernel_mul_mv_mma_" tname "_f32_nt1_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 1, bq, nl, deq>; \
template [[host_name("kernel_mul_mv_mma_" tname "_f32_nt2_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 1, bq, nl, deq>; \
template [[host_name("kernel_mul_mv_mma_" tname "_f32_nt4_rt1")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 1, bq, nl, deq>; \
template [[host_name("kernel_mul_mv_mma_" tname "_f32_nt1_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 2, bq, nl, deq>; \
template [[host_name("kernel_mul_mv_mma_" tname "_f32_nt2_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 2, bq, nl, deq>; \
template [[host_name("kernel_mul_mv_mma_" tname "_f32_nt4_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 2, bq, nl, deq>; \
template [[host_name("kernel_mul_mv_mma_id_" tname "_f32_nt4_rt1")]] kernel mul_mv_mma_id_t kernel_mul_mv_mma_id_gen<4, 1, bq, nl, deq>;

// q8_0 with 9..16 src1 rows: the per-block scaling above is slower than dequantizing to f32
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt1_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<1, 2, block_q8_0, 2, dequantize_q8_0>;
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt2_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<2, 2, block_q8_0, 2, dequantize_q8_0>;
template [[host_name("kernel_mul_mv_mma_q8_0_f32_nt4_rt2")]] kernel mul_mv_mma_t kernel_mul_mv_mma_gen<4, 2, block_q8_0, 2, dequantize_q8_0>;

MUL_MV_MMA_GEN("f32",  float4x4,   1,     dequantize_f32)
MUL_MV_MMA_GEN("f16",  half4x4,    1,     dequantize_f16)
MUL_MV_MMA_GEN("q4_1", block_q4_1, 2,     dequantize_q4_1)
MUL_MV_MMA_GEN("q5_0", block_q5_0, 2,     dequantize_q5_0)
MUL_MV_MMA_GEN("q5_1", block_q5_1, 2,     dequantize_q5_1)
MUL_MV_MMA_GEN("q4_K", block_q4_K, QK_NL, dequantize_q4_K)
MUL_MV_MMA_GEN("q6_K", block_q6_K, QK_NL, dequantize_q6_K)

#if defined(GGML_METAL_HAS_BF16)
MUL_MV_MMA_GEN("bf16",    bfloat4x4,     1,     dequantize_bf16)
#endif
MUL_MV_MMA_GEN("q1_0",    block_q1_0,    8,     dequantize_q1_0)
MUL_MV_MMA_GEN("q2_0",    block_q2_0,    4,     dequantize_q2_0)
MUL_MV_MMA_GEN("mxfp4",   block_mxfp4,   2,     dequantize_mxfp4)
MUL_MV_MMA_GEN("q2_K",    block_q2_K,    QK_NL, dequantize_q2_K)
MUL_MV_MMA_GEN("q3_K",    block_q3_K,    QK_NL, dequantize_q3_K)
MUL_MV_MMA_GEN("iq2_xxs", block_iq2_xxs, QK_NL, dequantize_iq2_xxs)
MUL_MV_MMA_GEN("iq2_xs",  block_iq2_xs,  QK_NL, dequantize_iq2_xs)
MUL_MV_MMA_GEN("iq2_s",   block_iq2_s,   QK_NL, dequantize_iq2_s)
MUL_MV_MMA_GEN("iq3_xxs", block_iq3_xxs, QK_NL, dequantize_iq3_xxs)
MUL_MV_MMA_GEN("iq3_s",   block_iq3_s,   QK_NL, dequantize_iq3_s)
MUL_MV_MMA_GEN("iq1_s",   block_iq1_s,   QK_NL, dequantize_iq1_s)
MUL_MV_MMA_GEN("iq1_m",   block_iq1_m,   QK_NL, dequantize_iq1_m)
MUL_MV_MMA_GEN("iq4_nl",  block_iq4_nl,  2,     dequantize_iq4_nl)
MUL_MV_MMA_GEN("iq4_xs",  block_iq4_xs,  QK_NL, dequantize_iq4_xs)
MUL_MV_MMA_GEN("tq2_0",   block_tq2_0,   QK_NL, dequantize_tq2_0)

#undef MUL_MV_MMA_GEN
