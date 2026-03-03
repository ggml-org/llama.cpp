enable f16;

#define Q_TILE 1
#define KV_TILE 32
#define WG_SIZE 32

struct Params {
    offset_mask: u32,
    seq_len_q: u32,
    seq_len_kv: u32,
    stride_mask3: u32,
    nblk0: u32,
    nblk1: u32,
};

@group(0) @binding(0) var<storage, read> mask: array<f16>;
@group(0) @binding(1) var<storage, read_write> blk: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const MASK_MIN: f32 = -65504.0;
const MASK_MAX: f32 = 65504.0;
var<workgroup> wg_min: array<f32, WG_SIZE>;
var<workgroup> wg_max: array<f32, WG_SIZE>;
var<workgroup> wg_any: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let kv_blk = wg_id.x;
    let y = wg_id.y;
    let q_blk = y % params.nblk1;
    let batch_idx = y / params.nblk1;
    if (kv_blk >= params.nblk0) {
        return;
    }

    let q_start = q_blk * Q_TILE;
    let k_start = kv_blk * KV_TILE;

    let mask_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
    let mask_batch_base = params.offset_mask + mask_batch * params.stride_mask3;

    var local_min = MASK_MAX;
    var local_max = -MASK_MAX;
    var local_any = 0u;

    for (var q_rel = 0u; q_rel < Q_TILE; q_rel += 1u) {
        let q_row = q_start + q_rel;
        if (q_row >= params.seq_len_q) {
            continue;
        }
        let row_base = mask_batch_base + q_row * params.seq_len_kv;
        for (var k_rel = local_id.x; k_rel < KV_TILE; k_rel += WG_SIZE) {
            let k_col = k_start + k_rel;
            if (k_col >= params.seq_len_kv) {
                continue;
            }
            let mv = f32(mask[row_base + k_col]);
            local_min = min(local_min, mv);
            local_max = max(local_max, mv);
            local_any = 1u;
        }
    }

    wg_min[local_id.x] = local_min;
    wg_max[local_id.x] = local_max;
    wg_any[local_id.x] = local_any;
    workgroupBarrier();

    if (local_id.x == 0u) {
        var mmin = wg_min[0];
        var mmax = wg_max[0];
        var many = wg_any[0];
        for (var i = 1u; i < WG_SIZE; i += 1u) {
            mmin = min(mmin, wg_min[i]);
            mmax = max(mmax, wg_max[i]);
            many = max(many, wg_any[i]);
        }

        var state = 0u;
        if (many != 0u) {
            if (mmax <= MASK_MIN) {
                state = 0u;
            } else if (mmin == 0.0 && mmax == 0.0) {
                state = 2u;
            } else {
                state = 1u;
            }
        }

        let blk_idx = (batch_idx * params.nblk1 + q_blk) * params.nblk0 + kv_blk;
        blk[blk_idx] = state;
    }
}
