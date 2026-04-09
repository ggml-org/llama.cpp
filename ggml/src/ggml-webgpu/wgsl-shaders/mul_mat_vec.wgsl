enable subgroups;
enable f16;

#include "common_decls.tmpl"

#ifdef U32_DEQUANT_HELPERS
#define SRC0_TYPE u32
fn byte_of(v: u32, b: u32) -> u32 {
    return (v >> (b * 8u)) & 0xFFu;
}

fn sbyte_of(v: u32, b: u32) -> i32 {
    let raw = i32((v >> (b * 8u)) & 0xFFu);
    return select(raw, raw - 256, raw >= 128);
}
#endif

#ifdef VEC

#define VEC_SIZE 4
#define DST_TYPE vec4<f32>
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(dot(SRC1_TYPE(src0_val), src1_val));
}

fn store_val(group_base: u32) -> vec4<f32> {
    return vec4<f32>(partial_sums[group_base],
                     partial_sums[group_base + THREADS_PER_OUTPUT],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 2],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 3]);
}
#endif

#ifdef SCALAR

#define VEC_SIZE 1
#define DST_TYPE f32
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(src0_val) * f32(src1_val);
}

fn store_val(group_base: u32) -> f32 {
    return partial_sums[group_base];
}
#endif

#ifdef MUL_ACC_FLOAT
fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * VEC_SIZE; i < tile_size; i += THREADS_PER_OUTPUT * VEC_SIZE) {
        let a = src0[(idx_base + k_outer + i) / VEC_SIZE];
        let b = shared_vector[i / VEC_SIZE];
        local_sum += inner_dot(a, b);
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q4_0

const BLOCK_SIZE = 32;
const BLOCK_SIZE_BYTES = 18u;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let block_byte_base = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * BLOCK_SIZE_BYTES;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(load_src0_f16_at(block_byte_base));
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_packed = load_src0_u32_at(block_byte_base + 2u + 2u * (block_offset + j));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0) * d;
                local_sum += q_lo * shared_vector[shmem_idx + j * 2 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 2 + k + 16];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q4_1

const BLOCK_SIZE = 32;
const BLOCK_SIZE_BYTES = 20u;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let block_byte_base = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * BLOCK_SIZE_BYTES;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(load_src0_f16_at(block_byte_base));
        let m = f32(load_src0_f16_at(block_byte_base + 2u));
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_packed = load_src0_u32_at(block_byte_base + 4u + 2u * (block_offset + j));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let q_hi = f32((q_byte >> 4) & 0xF) * d + m;
                let q_lo = f32(q_byte & 0xF) * d + m;
                local_sum += q_lo * shared_vector[shmem_idx + j * 2 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 2 + k + 16];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q5_0

const BLOCK_SIZE = 32;
const BLOCK_SIZE_BYTES = 22u;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let block_byte_base = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * BLOCK_SIZE_BYTES;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(load_src0_f16_at(block_byte_base));
        let qh_packed = load_src0_u32_at(block_byte_base + 2u);

        for (var j = 0u; j < 2; j++) {
            let q_packed = load_src0_u32_at(block_byte_base + 6u + 2u * (block_offset + j * 2u));
            let j_adjusted = j + (block_offset / 2u);
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                let q_hi = (f32(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
                let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                let q_lo = (f32((q_byte & 0xF) | qh_lo) - 16.0) * d;
                local_sum += q_lo * shared_vector[shmem_idx + j * 4 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 4 + k + 16];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q5_1

const BLOCK_SIZE = 32;
const BLOCK_SIZE_BYTES = 24u;
const NQ = 16u;
const WEIGHTS_PER_F16 = 4u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let block_byte_base = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * BLOCK_SIZE_BYTES;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(load_src0_f16_at(block_byte_base));
        let m = load_src0_f16_at(block_byte_base + 2u);
        let qh_packed = load_src0_u32_at(block_byte_base + 4u);

        for (var j = 0u; j < 2; j++) {
            let q_packed = load_src0_u32_at(block_byte_base + 8u + 2u * (block_offset + j * 2u));
            let j_adjusted = j + (block_offset / 2u);
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let qh_hi = (qh_packed >> (j_adjusted * 4 + k + 12)) & 0x10;
                let q_hi = f32(((q_byte >> 4) & 0xF) | qh_hi) * d + f32(m);
                let qh_lo = ((qh_packed >> (j_adjusted * 4 + k)) << 4) & 0x10;
                let q_lo = f32((q_byte & 0xF) | qh_lo) * d + f32(m);
                local_sum += q_lo * shared_vector[shmem_idx + j * 4 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 4 + k + 16];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q8_0

const BLOCK_SIZE = 32;
const BLOCK_SIZE_BYTES = 34u;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let block_byte_base = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * BLOCK_SIZE_BYTES;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(load_src0_f16_at(block_byte_base));
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_packed = load_src0_u32_at(block_byte_base + 2u + 2u * (block_offset + j));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte_i32(q_packed, k);
                let q_val = f32(q_byte) * d;
                local_sum += q_val * shared_vector[shmem_idx + j * 2 + k];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q8_1

const BLOCK_SIZE = 32;
const BLOCK_SIZE_BYTES = 36u;
const NQ = 16u;
const WEIGHTS_PER_F16 = 2u;
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let block_byte_base = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * BLOCK_SIZE_BYTES;
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(load_src0_f16_at(block_byte_base));
        let m = load_src0_f16_at(block_byte_base + 2u);
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_packed = load_src0_u32_at(block_byte_base + 4u + 2u * (block_offset + j));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte_i32(q_packed, k);
                let q_val = f32(q_byte) * d + f32(m);
                local_sum += q_val * shared_vector[shmem_idx + j * 2 + k];
            }
        }
    }
    return local_sum;
}
#endif

#ifdef MUL_ACC_Q2_K

const Q2K_BLOCK_SIZE     = 256u;
const Q2K_BLOCK_SIZE_BYTES = 84u;  // 42 f16s * 2

fn mul_acc(tig: u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    let ix = tig / 8u;
    let it = tig % 8u;
    let iq = it / 4u;
    let ir = it % 4u;
    let is = (8u * ir) / 16u;

    let nb            = tile_size / Q2K_BLOCK_SIZE;
    let k_block_start = k_outer / Q2K_BLOCK_SIZE;
    let y4_offset     = 128u * iq + 8u * ir;

    let sc0_byte = 8u * iq + is;
    let sc2_byte = 8u * iq + is + 2u;
    let sc4_byte = 8u * iq + is + 4u;
    let sc6_byte = 8u * iq + is + 6u;
    let qs_byte  = 16u + (16u * iq + 4u * ir) * 2u;

    var sumf = 0.0;

    for (var ib = ix; ib < nb; ib += 4u) {
        let bbase = (idx_base + k_block_start + ib) * Q2K_BLOCK_SIZE_BYTES;

        let dall = f32(load_src0_f16_at(bbase + 80u));
        let dmin = f32(load_src0_f16_at(bbase + 82u)) * (1.0 / 16.0);

        let sc0 = byte_of(load_src0_u32_at_aligned(bbase + sc0_byte), sc0_byte & 3u);
        let sc2 = byte_of(load_src0_u32_at_aligned(bbase + sc2_byte), sc2_byte & 3u);
        let sc4 = byte_of(load_src0_u32_at_aligned(bbase + sc4_byte), sc4_byte & 3u);
        let sc6 = byte_of(load_src0_u32_at_aligned(bbase + sc6_byte), sc6_byte & 3u);

        let qs_u32_0 = load_src0_u32_at_aligned(bbase + qs_byte);
        let qs_u32_1 = load_src0_u32_at_aligned(bbase + qs_byte + 4u);
        let qs0 = qs_u32_0 & 0xFFFFu;
        let qs1 = qs_u32_0 >> 16u;
        let qs2 = qs_u32_1 & 0xFFFFu;
        let qs3 = qs_u32_1 >> 16u;

        let y_base = ib * Q2K_BLOCK_SIZE + y4_offset;

        var sumy = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var acc1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var acc2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // i=0: j=0,1
        {
            let y00 = f32(shared_vector[y_base      ]); sumy[0] += y00;
            let y01 = f32(shared_vector[y_base +  1u]); sumy[0] += y01;
            let y10 = f32(shared_vector[y_base + 32u]); sumy[1] += y10;
            let y11 = f32(shared_vector[y_base + 33u]); sumy[1] += y11;
            let y20 = f32(shared_vector[y_base + 64u]); sumy[2] += y20;
            let y21 = f32(shared_vector[y_base + 65u]); sumy[2] += y21;
            let y30 = f32(shared_vector[y_base + 96u]); sumy[3] += y30;
            let y31 = f32(shared_vector[y_base + 97u]); sumy[3] += y31;
            acc1[0] += y00 * f32(qs0 & 0x0003u);
            acc2[0] += y01 * f32(qs0 & 0x0300u);
            acc1[1] += y10 * f32(qs0 & 0x000Cu);
            acc2[1] += y11 * f32(qs0 & 0x0C00u);
            acc1[2] += y20 * f32(qs0 & 0x0030u);
            acc2[2] += y21 * f32(qs0 & 0x3000u);
            acc1[3] += y30 * f32(qs0 & 0x00C0u);
            acc2[3] += y31 * f32(qs0 & 0xC000u);
        }
        // i=2: j=2,3
        {
            let y00 = f32(shared_vector[y_base +  2u]); sumy[0] += y00;
            let y01 = f32(shared_vector[y_base +  3u]); sumy[0] += y01;
            let y10 = f32(shared_vector[y_base + 34u]); sumy[1] += y10;
            let y11 = f32(shared_vector[y_base + 35u]); sumy[1] += y11;
            let y20 = f32(shared_vector[y_base + 66u]); sumy[2] += y20;
            let y21 = f32(shared_vector[y_base + 67u]); sumy[2] += y21;
            let y30 = f32(shared_vector[y_base + 98u]); sumy[3] += y30;
            let y31 = f32(shared_vector[y_base + 99u]); sumy[3] += y31;
            acc1[0] += y00 * f32(qs1 & 0x0003u);
            acc2[0] += y01 * f32(qs1 & 0x0300u);
            acc1[1] += y10 * f32(qs1 & 0x000Cu);
            acc2[1] += y11 * f32(qs1 & 0x0C00u);
            acc1[2] += y20 * f32(qs1 & 0x0030u);
            acc2[2] += y21 * f32(qs1 & 0x3000u);
            acc1[3] += y30 * f32(qs1 & 0x00C0u);
            acc2[3] += y31 * f32(qs1 & 0xC000u);
        }
        // i=4: j=4,5
        {
            let y00 = f32(shared_vector[y_base +  4u]); sumy[0] += y00;
            let y01 = f32(shared_vector[y_base +  5u]); sumy[0] += y01;
            let y10 = f32(shared_vector[y_base + 36u]); sumy[1] += y10;
            let y11 = f32(shared_vector[y_base + 37u]); sumy[1] += y11;
            let y20 = f32(shared_vector[y_base + 68u]); sumy[2] += y20;
            let y21 = f32(shared_vector[y_base + 69u]); sumy[2] += y21;
            let y30 = f32(shared_vector[y_base + 100u]); sumy[3] += y30;
            let y31 = f32(shared_vector[y_base + 101u]); sumy[3] += y31;
            acc1[0] += y00 * f32(qs2 & 0x0003u);
            acc2[0] += y01 * f32(qs2 & 0x0300u);
            acc1[1] += y10 * f32(qs2 & 0x000Cu);
            acc2[1] += y11 * f32(qs2 & 0x0C00u);
            acc1[2] += y20 * f32(qs2 & 0x0030u);
            acc2[2] += y21 * f32(qs2 & 0x3000u);
            acc1[3] += y30 * f32(qs2 & 0x00C0u);
            acc2[3] += y31 * f32(qs2 & 0xC000u);
        }
        // i=6: j=6,7
        {
            let y00 = f32(shared_vector[y_base +  6u]); sumy[0] += y00;
            let y01 = f32(shared_vector[y_base +  7u]); sumy[0] += y01;
            let y10 = f32(shared_vector[y_base + 38u]); sumy[1] += y10;
            let y11 = f32(shared_vector[y_base + 39u]); sumy[1] += y11;
            let y20 = f32(shared_vector[y_base + 70u]); sumy[2] += y20;
            let y21 = f32(shared_vector[y_base + 71u]); sumy[2] += y21;
            let y30 = f32(shared_vector[y_base + 102u]); sumy[3] += y30;
            let y31 = f32(shared_vector[y_base + 103u]); sumy[3] += y31;
            acc1[0] += y00 * f32(qs3 & 0x0003u);
            acc2[0] += y01 * f32(qs3 & 0x0300u);
            acc1[1] += y10 * f32(qs3 & 0x000Cu);
            acc2[1] += y11 * f32(qs3 & 0x0C00u);
            acc1[2] += y20 * f32(qs3 & 0x0030u);
            acc2[2] += y21 * f32(qs3 & 0x3000u);
            acc1[3] += y30 * f32(qs3 & 0x00C0u);
            acc2[3] += y31 * f32(qs3 & 0xC000u);
        }

        sumf += dall * ((acc1[0] + (1.0/256.0) * acc2[0]) * f32(sc0 & 0xFu)        +
                        (acc1[1] + (1.0/256.0) * acc2[1]) * f32(sc2 & 0xFu) / 4.0  +
                        (acc1[2] + (1.0/256.0) * acc2[2]) * f32(sc4 & 0xFu) / 16.0 +
                        (acc1[3] + (1.0/256.0) * acc2[3]) * f32(sc6 & 0xFu) / 64.0)
             - dmin * (sumy[0] * f32(sc0 & 0xF0u) + sumy[1] * f32(sc2 & 0xF0u) +
                       sumy[2] * f32(sc4 & 0xF0u) + sumy[3] * f32(sc6 & 0xF0u));
    }

    return sumf;
}
#endif

#ifdef MUL_ACC_Q3_K

const Q3K_BLOCK_SIZE       = 256u;
const Q3K_BLOCK_SIZE_BYTES = 110u;  // 55 f16s * 2

fn mul_acc(tig: u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    let tid = tig / 4u;
    let ix  = tig % 4u;
    let ip  = tid / 4u;
    let il  = 2u * ((tid % 4u) / 2u);
    let ir  = tid % 2u;
    let l0  = 8u * ir;

    let nb            = tile_size / Q3K_BLOCK_SIZE;
    let k_block_start = k_outer / Q3K_BLOCK_SIZE;

    let q_byte   = 32u + 32u * ip + l0;
    let h_byte   = l0;
    let y_offset = 128u * ip + 32u * il + l0;

    let s_shift1 = 4u * ip;
    let s_shift2 = s_shift1 + il;

    let v1 = select(64.0, 4.0, il == 0u);
    let v2 = 4.0 * v1;
    let shift = 2u * il;

    var qm0: u32; var qm1: u32; var qm2: u32; var qm3: u32;
    if (il == 0u) {
        qm0 = 0x0003u; qm1 = 0x0300u; qm2 = 0x000Cu; qm3 = 0x0C00u;
    } else {
        qm0 = 0x0030u; qm1 = 0x3000u; qm2 = 0x00C0u; qm3 = 0xC000u;
    }

    let mm_idx = 2u * ip + il / 2u;
    var hm0: u32; var hm1: u32; var hm2: u32; var hm3: u32;
    switch (mm_idx) {
        case 0u: { hm0=0x0001u; hm1=0x0100u; hm2=0x0002u; hm3=0x0200u; }
        case 1u: { hm0=0x0004u; hm1=0x0400u; hm2=0x0008u; hm3=0x0800u; }
        case 2u: { hm0=0x0010u; hm1=0x1000u; hm2=0x0020u; hm3=0x2000u; }
        default: { hm0=0x0040u; hm1=0x4000u; hm2=0x0080u; hm3=0x8000u; }
    }

    var sumf1 = 0.0;
    var sumf2 = 0.0;

    for (var i = ix; i < nb; i += 4u) {
        let bbase = (idx_base + k_block_start + i) * Q3K_BLOCK_SIZE_BYTES;

        let d_all = f32(load_src0_f16_at(bbase + 108u));

        // Scale unpacking
        let a_base = 96u;
        let a_il0_u32 = load_src0_u32_at_aligned(bbase + a_base + il * 2u);
        let a_il0 = select(a_il0_u32 & 0xFFFFu, a_il0_u32 >> 16u, (il & 1u) != 0u);
        let a_il1_u32 = load_src0_u32_at_aligned(bbase + a_base + (il + 1u) * 2u);
        let a_il1 = select(a_il1_u32 & 0xFFFFu, a_il1_u32 >> 16u, ((il + 1u) & 1u) != 0u);
        let a_45_u32 = load_src0_u32_at_aligned(bbase + a_base + 8u);
        let a_4 = a_45_u32 & 0xFFFFu;
        let a_5 = a_45_u32 >> 16u;

        var scales32 = a_4 | (a_5 << 16u);
        let aux32 = ((scales32 >> s_shift2) << 4u) & 0x30303030u;
        scales32 = a_il0 | (a_il1 << 16u);
        scales32 = ((scales32 >> s_shift1) & 0x0F0F0F0Fu) | aux32;

        let sc0 = i32(byte_of(scales32, 0u)) - 32;
        let sc1 = i32(byte_of(scales32, 1u)) - 32;
        let sc2 = i32(byte_of(scales32, 2u)) - 32;
        let sc3 = i32(byte_of(scales32, 3u)) - 32;

        let y_base = i * Q3K_BLOCK_SIZE + y_offset;
        var yl: array<f32, 32>;
        for (var l = 0u; l < 8u; l++) {
            yl[l +  0] = f32(shared_vector[y_base + l      ]);
            yl[l +  8] = f32(shared_vector[y_base + l + 16u]);
            yl[l + 16] = f32(shared_vector[y_base + l + 32u]);
            yl[l + 24] = f32(shared_vector[y_base + l + 48u]);
        }

        // First qs/h loop: q[0..3], h[0..3]
        var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
        var s4 = 0.0; var s5 = 0.0; var s6 = 0.0;

        for (var l = 0u; l < 8u; l += 2u) {
            let q_u32 = load_src0_u32_at_aligned(bbase + q_byte + (l & ~2u));
            let qs    = select(q_u32 & 0xFFFFu, q_u32 >> 16u, (l & 2u) != 0u);
            let h_u32 = load_src0_u32_at_aligned(bbase + h_byte + (l & ~2u));
            let hv    = select(h_u32 & 0xFFFFu, h_u32 >> 16u, (l & 2u) != 0u);

            s1 += yl[l + 0u] * f32(qs & qm0);
            s2 += yl[l + 1u] * f32(qs & qm1);
            s3 += select(0.0, yl[l + 0u], (hv & hm0) == 0u) +
                  select(0.0, yl[l + 1u], (hv & hm1) == 0u);
            s4 += yl[l + 16u] * f32(qs & qm2);
            s5 += yl[l + 17u] * f32(qs & qm3);
            s6 += select(0.0, yl[l + 16u], (hv & hm2) == 0u) +
                  select(0.0, yl[l + 17u], (hv & hm3) == 0u);
        }
        let d1 = d_all * (s1 + (1.0/256.0)*s2 - s3*v1);
        let d2 = d_all * (s4 + (1.0/256.0)*s5 - s6*v2);
        sumf1 += d1 * f32(sc0);
        sumf2 += d2 * f32(sc2);

        // Second qs/h loop: q[8..11], h[8..11] (16 bytes further)
        s1 = 0.0; s2 = 0.0; s3 = 0.0;
        s4 = 0.0; s5 = 0.0; s6 = 0.0;

        for (var l = 0u; l < 8u; l += 2u) {
            let q_u32 = load_src0_u32_at_aligned(bbase + q_byte + 16u + (l & ~2u));
            let qs    = select(q_u32 & 0xFFFFu, q_u32 >> 16u, (l & 2u) != 0u);
            let h_u32 = load_src0_u32_at_aligned(bbase + h_byte + 16u + (l & ~2u));
            let hv    = select(h_u32 & 0xFFFFu, h_u32 >> 16u, (l & 2u) != 0u);

            s1 += yl[l +  8u] * f32(qs & qm0);
            s2 += yl[l +  9u] * f32(qs & qm1);
            s3 += select(0.0, yl[l +  8u], (hv & hm0) == 0u) +
                  select(0.0, yl[l +  9u], (hv & hm1) == 0u);
            s4 += yl[l + 24u] * f32(qs & qm2);
            s5 += yl[l + 25u] * f32(qs & qm3);
            s6 += select(0.0, yl[l + 24u], (hv & hm2) == 0u) +
                  select(0.0, yl[l + 25u], (hv & hm3) == 0u);
        }
        let d3 = d_all * (s1 + (1.0/256.0)*s2 - s3*v1);
        let d4 = d_all * (s4 + (1.0/256.0)*s5 - s6*v2);
        sumf1 += d3 * f32(sc1);
        sumf2 += d4 * f32(sc3);
    }

    return (sumf1 + 0.25 * sumf2) / f32(1u << shift);
}
#endif

#ifdef MUL_ACC_Q4_K

const Q4K_BLOCK_SIZE       = 256u;
const Q4K_BLOCK_SIZE_BYTES = 144u;  // 72 f16s * 2

fn mul_acc(tig: u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    let ix = tig / 8u;
    let it = tig % 8u;
    let iq = it / 4u;
    let ir = it % 4u;

    let nb            = tile_size / Q4K_BLOCK_SIZE;
    let k_block_start = k_outer / Q4K_BLOCK_SIZE;

    let y_offset = 64u * iq + 8u * ir;

    let sc0_byte = 4u + iq * 2u;
    let sc2_byte = 4u + (iq + 2u) * 2u;
    let sc4_byte = 4u + (iq + 4u) * 2u;
    let q1_byte  = 16u + (16u * iq + 4u * ir) * 2u;
    let q2_byte  = q1_byte + 64u;

    var sumf = 0.0;

    for (var ib = ix; ib < nb; ib += 4u) {
        let bbase = (idx_base + k_block_start + ib) * Q4K_BLOCK_SIZE_BYTES;

        let d    = f32(load_src0_f16_at(bbase + 0u));
        let dmin = f32(load_src0_f16_at(bbase + 2u));

        let sc0_u32 = load_src0_u32_at_aligned(bbase + sc0_byte);
        let sc0 = select(sc0_u32 & 0xFFFFu, sc0_u32 >> 16u, (sc0_byte & 2u) != 0u);
        let sc2_u32 = load_src0_u32_at_aligned(bbase + sc2_byte);
        let sc2 = select(sc2_u32 & 0xFFFFu, sc2_u32 >> 16u, (sc2_byte & 2u) != 0u);
        let sc4_u32 = load_src0_u32_at_aligned(bbase + sc4_byte);
        let sc4 = select(sc4_u32 & 0xFFFFu, sc4_u32 >> 16u, (sc4_byte & 2u) != 0u);

        let sc16_0 =  sc0 & 0x3F3Fu;
        let sc16_1 =  sc2 & 0x3F3Fu;
        let sc16_2 = ((sc4        ) & 0x0F0Fu) | ((sc0 & 0xC0C0u) >> 2u);
        let sc16_3 = ((sc4 >>  4u ) & 0x0F0Fu) | ((sc2 & 0xC0C0u) >> 2u);

        let sc8_0 =  sc16_0 & 0xFFu;
        let sc8_1 = (sc16_0 >> 8u) & 0xFFu;
        let sc8_2 =  sc16_1 & 0xFFu;
        let sc8_3 = (sc16_1 >> 8u) & 0xFFu;
        let sc8_4 =  sc16_2 & 0xFFu;
        let sc8_5 = (sc16_2 >> 8u) & 0xFFu;
        let sc8_6 =  sc16_3 & 0xFFu;
        let sc8_7 = (sc16_3 >> 8u) & 0xFFu;

        let q1_u32_0 = load_src0_u32_at_aligned(bbase + q1_byte);
        let q1_u32_1 = load_src0_u32_at_aligned(bbase + q1_byte + 4u);
        let q2_u32_0 = load_src0_u32_at_aligned(bbase + q2_byte);
        let q2_u32_1 = load_src0_u32_at_aligned(bbase + q2_byte + 4u);

        let q1_0 = q1_u32_0 & 0xFFFFu;
        let q1_1 = q1_u32_0 >> 16u;
        let q1_2 = q1_u32_1 & 0xFFFFu;
        let q1_3 = q1_u32_1 >> 16u;
        let q2_0 = q2_u32_0 & 0xFFFFu;
        let q2_1 = q2_u32_0 >> 16u;
        let q2_2 = q2_u32_1 & 0xFFFFu;
        let q2_3 = q2_u32_1 >> 16u;

        let y_base = ib * Q4K_BLOCK_SIZE + y_offset;

        var sumy = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var acc1  = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var acc2  = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // i=0: yl[0,1,8,9], yh[0,1,8,9]
        {
            let yl0 = f32(shared_vector[y_base +   0u]); sumy[0] += yl0;
            let yl1 = f32(shared_vector[y_base +   1u]); sumy[0] += yl1;
            let yl8 = f32(shared_vector[y_base +  32u]); sumy[1] += yl8;
            let yl9 = f32(shared_vector[y_base +  33u]); sumy[1] += yl9;
            let yh0 = f32(shared_vector[y_base + 128u]); sumy[2] += yh0;
            let yh1 = f32(shared_vector[y_base + 129u]); sumy[2] += yh1;
            let yh8 = f32(shared_vector[y_base + 160u]); sumy[3] += yh8;
            let yh9 = f32(shared_vector[y_base + 161u]); sumy[3] += yh9;
            acc1[0] += yl0 * f32(q1_0 & 0x000Fu);
            acc1[1] += yl1 * f32(q1_0 & 0x0F00u);
            acc1[2] += yl8 * f32(q1_0 & 0x00F0u);
            acc1[3] += yl9 * f32(q1_0 & 0xF000u);
            acc2[0] += yh0 * f32(q2_0 & 0x000Fu);
            acc2[1] += yh1 * f32(q2_0 & 0x0F00u);
            acc2[2] += yh8 * f32(q2_0 & 0x00F0u);
            acc2[3] += yh9 * f32(q2_0 & 0xF000u);
        }
        // i=1: yl[2,3,10,11], yh[2,3,10,11]
        {
            let yl0 = f32(shared_vector[y_base +   2u]); sumy[0] += yl0;
            let yl1 = f32(shared_vector[y_base +   3u]); sumy[0] += yl1;
            let yl8 = f32(shared_vector[y_base +  34u]); sumy[1] += yl8;
            let yl9 = f32(shared_vector[y_base +  35u]); sumy[1] += yl9;
            let yh0 = f32(shared_vector[y_base + 130u]); sumy[2] += yh0;
            let yh1 = f32(shared_vector[y_base + 131u]); sumy[2] += yh1;
            let yh8 = f32(shared_vector[y_base + 162u]); sumy[3] += yh8;
            let yh9 = f32(shared_vector[y_base + 163u]); sumy[3] += yh9;
            acc1[0] += yl0 * f32(q1_1 & 0x000Fu);
            acc1[1] += yl1 * f32(q1_1 & 0x0F00u);
            acc1[2] += yl8 * f32(q1_1 & 0x00F0u);
            acc1[3] += yl9 * f32(q1_1 & 0xF000u);
            acc2[0] += yh0 * f32(q2_1 & 0x000Fu);
            acc2[1] += yh1 * f32(q2_1 & 0x0F00u);
            acc2[2] += yh8 * f32(q2_1 & 0x00F0u);
            acc2[3] += yh9 * f32(q2_1 & 0xF000u);
        }
        // i=2: yl[4,5,12,13], yh[4,5,12,13]
        {
            let yl0 = f32(shared_vector[y_base +   4u]); sumy[0] += yl0;
            let yl1 = f32(shared_vector[y_base +   5u]); sumy[0] += yl1;
            let yl8 = f32(shared_vector[y_base +  36u]); sumy[1] += yl8;
            let yl9 = f32(shared_vector[y_base +  37u]); sumy[1] += yl9;
            let yh0 = f32(shared_vector[y_base + 132u]); sumy[2] += yh0;
            let yh1 = f32(shared_vector[y_base + 133u]); sumy[2] += yh1;
            let yh8 = f32(shared_vector[y_base + 164u]); sumy[3] += yh8;
            let yh9 = f32(shared_vector[y_base + 165u]); sumy[3] += yh9;
            acc1[0] += yl0 * f32(q1_2 & 0x000Fu);
            acc1[1] += yl1 * f32(q1_2 & 0x0F00u);
            acc1[2] += yl8 * f32(q1_2 & 0x00F0u);
            acc1[3] += yl9 * f32(q1_2 & 0xF000u);
            acc2[0] += yh0 * f32(q2_2 & 0x000Fu);
            acc2[1] += yh1 * f32(q2_2 & 0x0F00u);
            acc2[2] += yh8 * f32(q2_2 & 0x00F0u);
            acc2[3] += yh9 * f32(q2_2 & 0xF000u);
        }
        // i=3: yl[6,7,14,15], yh[6,7,14,15]
        {
            let yl0 = f32(shared_vector[y_base +   6u]); sumy[0] += yl0;
            let yl1 = f32(shared_vector[y_base +   7u]); sumy[0] += yl1;
            let yl8 = f32(shared_vector[y_base +  38u]); sumy[1] += yl8;
            let yl9 = f32(shared_vector[y_base +  39u]); sumy[1] += yl9;
            let yh0 = f32(shared_vector[y_base + 134u]); sumy[2] += yh0;
            let yh1 = f32(shared_vector[y_base + 135u]); sumy[2] += yh1;
            let yh8 = f32(shared_vector[y_base + 166u]); sumy[3] += yh8;
            let yh9 = f32(shared_vector[y_base + 167u]); sumy[3] += yh9;
            acc1[0] += yl0 * f32(q1_3 & 0x000Fu);
            acc1[1] += yl1 * f32(q1_3 & 0x0F00u);
            acc1[2] += yl8 * f32(q1_3 & 0x00F0u);
            acc1[3] += yl9 * f32(q1_3 & 0xF000u);
            acc2[0] += yh0 * f32(q2_3 & 0x000Fu);
            acc2[1] += yh1 * f32(q2_3 & 0x0F00u);
            acc2[2] += yh8 * f32(q2_3 & 0x00F0u);
            acc2[3] += yh9 * f32(q2_3 & 0xF000u);
        }

        sumf += d * ((acc1[0] + (1.0/256.0)*acc1[1]) * f32(sc8_0)              +
                     (acc1[2] + (1.0/256.0)*acc1[3]) * f32(sc8_1) * (1.0/16.0) +
                     (acc2[0] + (1.0/256.0)*acc2[1]) * f32(sc8_4)              +
                     (acc2[2] + (1.0/256.0)*acc2[3]) * f32(sc8_5) * (1.0/16.0))
             - dmin * (sumy[0] * f32(sc8_2) + sumy[1] * f32(sc8_3) +
                       sumy[2] * f32(sc8_6) + sumy[3] * f32(sc8_7));
    }

    return sumf;
}
#endif

#ifdef MUL_ACC_Q5_K

const Q5K_BLOCK_SIZE       = 256u;
const Q5K_BLOCK_SIZE_BYTES = 176u;  // 88 f16s * 2

fn mul_acc(tig: u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    let tid = tig / 4u;
    let ix  = tig % 4u;
    let iq  = tid / 4u;
    let ir  = tid % 4u;
    let l0  = 8u * ir;

    let nb            = tile_size / Q5K_BLOCK_SIZE;
    let k_block_start = k_outer / Q5K_BLOCK_SIZE;

    let y_offset = 64u * iq + l0;
    let q1_byte  = 48u + 32u * iq + l0;
    let q2_byte  = q1_byte + 64u;
    let qh_byte  = 16u + l0;
    let sc0_byte = 4u + iq * 2u;
    let sc2_byte = 4u + (iq + 2u) * 2u;
    let sc4_byte = 4u + (iq + 4u) * 2u;

    let hm1 = 1u << (2u * iq);
    let hm2 = hm1 << 1u;
    let hm3 = hm1 << 4u;
    let hm4 = hm2 << 4u;

    var sumf = 0.0;

    for (var ib = ix; ib < nb; ib += 4u) {
        let bbase = (idx_base + k_block_start + ib) * Q5K_BLOCK_SIZE_BYTES;

        let d    = f32(load_src0_f16_at(bbase + 0u));
        let dmin = f32(load_src0_f16_at(bbase + 2u));

        let sc0_u32 = load_src0_u32_at_aligned(bbase + sc0_byte);
        let sc0 = select(sc0_u32 & 0xFFFFu, sc0_u32 >> 16u, (sc0_byte & 2u) != 0u);
        let sc2_u32 = load_src0_u32_at_aligned(bbase + sc2_byte);
        let sc2 = select(sc2_u32 & 0xFFFFu, sc2_u32 >> 16u, (sc2_byte & 2u) != 0u);
        let sc4_u32 = load_src0_u32_at_aligned(bbase + sc4_byte);
        let sc4 = select(sc4_u32 & 0xFFFFu, sc4_u32 >> 16u, (sc4_byte & 2u) != 0u);

        let sc16_0 =  sc0 & 0x3F3Fu;
        let sc16_1 =  sc2 & 0x3F3Fu;
        let sc16_2 = ((sc4        ) & 0x0F0Fu) | ((sc0 & 0xC0C0u) >> 2u);
        let sc16_3 = ((sc4 >>  4u ) & 0x0F0Fu) | ((sc2 & 0xC0C0u) >> 2u);

        let sc8_0 =  sc16_0 & 0xFFu;
        let sc8_1 = (sc16_0 >> 8u) & 0xFFu;
        let sc8_2 =  sc16_1 & 0xFFu;
        let sc8_3 = (sc16_1 >> 8u) & 0xFFu;
        let sc8_4 =  sc16_2 & 0xFFu;
        let sc8_5 = (sc16_2 >> 8u) & 0xFFu;
        let sc8_6 =  sc16_3 & 0xFFu;
        let sc8_7 = (sc16_3 >> 8u) & 0xFFu;

        let f0    = f32(sc8_0);
        let f1_lo = f32(sc8_1) * (1.0/16.0);
        let f1_hi = f32(sc8_1) * 16.0;
        let f4    = f32(sc8_4);
        let f5_lo = f32(sc8_5) * (1.0/16.0);
        let f5_hi = f32(sc8_5) * 16.0;

        let q1_u32_0 = load_src0_u32_at_aligned(bbase + q1_byte);
        let q1_u32_1 = load_src0_u32_at_aligned(bbase + q1_byte + 4u);
        let q2_u32_0 = load_src0_u32_at_aligned(bbase + q2_byte);
        let q2_u32_1 = load_src0_u32_at_aligned(bbase + q2_byte + 4u);
        let qh_u32_0 = load_src0_u32_at_aligned(bbase + qh_byte);
        let qh_u32_1 = load_src0_u32_at_aligned(bbase + qh_byte + 4u);

        let y_base = ib * Q5K_BLOCK_SIZE + y_offset;

        var sumy = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var acc1  = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var acc2  = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        // l=0
        {
            let q1b = byte_of(q1_u32_0, 0u);
            let q2b = byte_of(q2_u32_0, 0u);
            let qhb = byte_of(qh_u32_0, 0u);
            let yl0 = f32(shared_vector[y_base +  0u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 32u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +128u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +160u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=1
        {
            let q1b = byte_of(q1_u32_0, 1u);
            let q2b = byte_of(q2_u32_0, 1u);
            let qhb = byte_of(qh_u32_0, 1u);
            let yl0 = f32(shared_vector[y_base +  1u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 33u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +129u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +161u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=2
        {
            let q1b = byte_of(q1_u32_0, 2u);
            let q2b = byte_of(q2_u32_0, 2u);
            let qhb = byte_of(qh_u32_0, 2u);
            let yl0 = f32(shared_vector[y_base +  2u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 34u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +130u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +162u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=3
        {
            let q1b = byte_of(q1_u32_0, 3u);
            let q2b = byte_of(q2_u32_0, 3u);
            let qhb = byte_of(qh_u32_0, 3u);
            let yl0 = f32(shared_vector[y_base +  3u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 35u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +131u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +163u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=4
        {
            let q1b = byte_of(q1_u32_1, 0u);
            let q2b = byte_of(q2_u32_1, 0u);
            let qhb = byte_of(qh_u32_1, 0u);
            let yl0 = f32(shared_vector[y_base +  4u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 36u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +132u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +164u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=5
        {
            let q1b = byte_of(q1_u32_1, 1u);
            let q2b = byte_of(q2_u32_1, 1u);
            let qhb = byte_of(qh_u32_1, 1u);
            let yl0 = f32(shared_vector[y_base +  5u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 37u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +133u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +165u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=6
        {
            let q1b = byte_of(q1_u32_1, 2u);
            let q2b = byte_of(q2_u32_1, 2u);
            let qhb = byte_of(qh_u32_1, 2u);
            let yl0 = f32(shared_vector[y_base +  6u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 38u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +134u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +166u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }
        // l=7
        {
            let q1b = byte_of(q1_u32_1, 3u);
            let q2b = byte_of(q2_u32_1, 3u);
            let qhb = byte_of(qh_u32_1, 3u);
            let yl0 = f32(shared_vector[y_base +  7u]); sumy[0] += yl0;
            let yl8 = f32(shared_vector[y_base + 39u]); sumy[1] += yl8;
            let yh0 = f32(shared_vector[y_base +135u]); sumy[2] += yh0;
            let yh8 = f32(shared_vector[y_base +167u]); sumy[3] += yh8;
            acc1[0] += yl0 * f32(q1b & 0x0Fu);
            acc1[1] += yl8 * f32(q1b & 0xF0u);
            acc1[2] += yh0 * f32(q2b & 0x0Fu);
            acc1[3] += yh8 * f32(q2b & 0xF0u);
            acc2[0] += yl0 * f32((qhb & hm1) != 0u);
            acc2[1] += yl8 * f32((qhb & hm2) != 0u);
            acc2[2] += yh0 * f32((qhb & hm3) != 0u);
            acc2[3] += yh8 * f32((qhb & hm4) != 0u);
        }

        sumf += d    * (f0    * acc1[0] + f0    * 16.0 * acc2[0] +
                        f1_lo * acc1[1] + f1_hi *        acc2[1] +
                        f4    * acc1[2] + f4    * 16.0 * acc2[2] +
                        f5_lo * acc1[3] + f5_hi *        acc2[3])
              - dmin * (sumy[0]*f32(sc8_2) + sumy[1]*f32(sc8_3) +
                        sumy[2]*f32(sc8_6) + sumy[3]*f32(sc8_7));
    }

    return sumf;
}
#endif

#ifdef MUL_ACC_Q6_K

const BLOCK_SIZE           = 256u;
const BLOCK_SIZE_BYTES     = 210u;  // 105 f16s * 2

fn mul_acc(tig: u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    let tid = tig / 2u;
    let ix  = tig % 2u;
    let ip  = tid / 8u;
    let il  = tid % 8u;
    let l0  = 4u * il;
    let is  = 8u * ip + l0 / 16u;

    let y_offset   = 128u * ip + l0;
    let q_offset_l =  64u * ip + l0;
    let q_offset_h =  32u * ip + l0;

    let nb = tile_size / BLOCK_SIZE;
    let k_block_start = k_outer / BLOCK_SIZE;

    // Aligned scale byte position (is can be odd)
    let sc_base_byte = 192u + (is & ~3u);
    let sc_byte_pos  = is & 3u;

    var local_sum = 0.0;

    for (var i = ix; i < nb; i += 2u) {
        let bbase = (idx_base + k_block_start + i) * BLOCK_SIZE_BYTES;

        let d = f32(load_src0_f16_at(bbase + 208u));

        let ql1_u32  = load_src0_u32_at(bbase + q_offset_l);
        let ql2_u32  = load_src0_u32_at(bbase + q_offset_l + 32u);
        let qh_u32   = load_src0_u32_at(bbase + 128u + q_offset_h);
        let sc_u32_0 = load_src0_u32_at(bbase + sc_base_byte);
        let sc_u32_1 = load_src0_u32_at(bbase + sc_base_byte + 4u);

        let sc0 = sbyte_of(sc_u32_0, sc_byte_pos);
        let sc2 = sbyte_of(sc_u32_0, sc_byte_pos + 2u);
        let sc4 = sbyte_of(sc_u32_1, sc_byte_pos);
        let sc6 = sbyte_of(sc_u32_1, sc_byte_pos + 2u);

        var sums = vec4<f32>(0.0, 0.0, 0.0, 0.0);

        for (var l = 0u; l < 4u; l++) {
            let y_base = i * BLOCK_SIZE + y_offset + l;
            let yl0 = f32(shared_vector[y_base]);
            let yl1 = f32(shared_vector[y_base + 32u]);
            let yl2 = f32(shared_vector[y_base + 64u]);
            let yl3 = f32(shared_vector[y_base + 96u]);

            let q1b = byte_of(ql1_u32, l);
            let q2b = byte_of(ql2_u32, l);
            let qhb = byte_of(qh_u32,  l);

            let dq0 = f32(i32((q1b & 0x0Fu) | ((qhb & 0x03u) << 4u)) - 32);
            let dq1 = f32(i32((q2b & 0x0Fu) | ((qhb & 0x0Cu) << 2u)) - 32);
            let dq2 = f32(i32((q1b >>   4u) | ((qhb & 0x30u)       )) - 32);
            let dq3 = f32(i32((q2b >>   4u) | ((qhb & 0xC0u) >> 2u)) - 32);

            sums[0] += yl0 * dq0;
            sums[1] += yl1 * dq1;
            sums[2] += yl2 * dq2;
            sums[3] += yl3 * dq3;
        }

        local_sum += d * (sums[0] * f32(sc0) + sums[1] * f32(sc2) +
                          sums[2] * f32(sc4) + sums[3] * f32(sc6));
    }

    return local_sum;
}
#endif

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

// SRC0_TYPE and SRC1_TYPE are defined above (VEC/SCALAR for float, U32_DEQUANT_HELPERS for quantized)
@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>;   // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

const THREADS_PER_OUTPUT = WG_SIZE / OUTPUTS_PER_WG;
// const THREADS_PER_OUTPUT = 32u;

// Shared memory for collaborative loading and reduction
// padded by + 1 to serialize reads (perf improvement for legacy quants?)
var<workgroup> shared_vector: array<SRC1_TYPE, TILE_K/VEC_SIZE + 1>;  // Cache vector tile
var<workgroup> partial_sums: array<f32, WG_SIZE>;   // For reduction

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32) {
    let thread_id = local_id.x;

    // Handle batch dimensions
    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    // Which of the outputs does this thread belong to?
    let thread_group = thread_id / THREADS_PER_OUTPUT;
    let thread_in_group = thread_id % THREADS_PER_OUTPUT;

    // Each workgroup computes OUTPUTS_PER_WG consecutive outputs
    let output_row = (wg_linear % output_groups) * OUTPUTS_PER_WG + thread_group;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + output_row * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;
    let dst_idx = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + output_row;

    var local_sum = 0.0;

    // Each thread processes multiple K elements and accumulates
    for (var k_tile = 0u; k_tile < params.k; k_tile += TILE_K) {
        let tile_size = min(TILE_K, params.k - k_tile);

        // Cooperatively load vector tile into shared memory (all threads)
        for (var i = thread_id * VEC_SIZE; i < tile_size; i += WG_SIZE * VEC_SIZE) {
            shared_vector[i / VEC_SIZE] = src1[(src1_idx_base + k_tile + i) / VEC_SIZE];
        }

        workgroupBarrier();

        if (output_row < params.m) {
            local_sum += mul_acc(thread_in_group, tile_size, src0_idx_base, k_tile);
        }

        workgroupBarrier();
    }

    let subgroup_local_id = thread_in_group % subgroup_size;
    let subgroup_in_group = thread_in_group / subgroup_size;
    let subgroup_total    = subgroupAdd(local_sum);

    if (subgroup_local_id == 0u) {
        partial_sums[thread_group * THREADS_PER_OUTPUT + subgroup_in_group] = subgroup_total;
    }
    workgroupBarrier();

    if (thread_in_group == 0u && output_row < params.m) {
        var row_total = 0.0;
        let num_subgroups = THREADS_PER_OUTPUT / subgroup_size;
        for (var s = 0u; s < num_subgroups; s++) {
            row_total += partial_sums[thread_group * THREADS_PER_OUTPUT + s];
        }
        #ifdef SCALAR
            dst[dst_idx] = row_total;
        #endif
        #ifdef VEC
            partial_sums[thread_group * THREADS_PER_OUTPUT] = row_total;
        #endif
    }

    #ifdef VEC
    workgroupBarrier();
    if (output_row < params.m && thread_group % VEC_SIZE == 0u && thread_in_group == 0u) {
        dst[dst_idx / VEC_SIZE] = store_val(thread_group * THREADS_PER_OUTPUT);
    }
    #endif
}