@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> idx_in: array<i32>;

@group(0) @binding(2)
var<storage, read_write> idx_out: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_in: u32,  // in elements
    offset_out: u32, // in elements

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx1: u32,
    stride_idx2: u32,
    stride_idx3: u32,

    stride_out1: u32,
    stride_out2: u32,
    stride_out3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,

    top_k: u32,

    len: u32,
    nm: u32,
    nrows: u32
};

@group(0) @binding(3)
var<uniform> params: Params;

fn take_left(a_idx: i32, b_idx: i32, row_base: u32) -> bool {
    let a_val = src[row_base + u32(a_idx)];
    let b_val = src[row_base + u32(b_idx)];
#if ORDER == 0
    return a_val <= b_val;
#else
    return a_val >= b_val;
#endif
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let linear = wid.x + wid.y * num_wg.x;
    // guard against overprovisioned workgroups
    if (linear >= params.nm * params.nrows) {
        return;
    }

    let start = (linear % params.nm) * params.len * 2;
    let len0 = min(params.len, params.ne0 - start);
    let rem1 = select(0, params.ne0 - (start + params.len), params.ne0 > (start + params.len));
    let len1 = min(params.len, rem1);
    let total = len0 + len1;
    let chunk = (total + WG_SIZE - 1u) / WG_SIZE;
    let k0 = lid.x * chunk;
    let k1 = min(min(k0 + chunk, total), params.top_k);
    // guard against overprovisioned threads
    if (k0 >= params.top_k || k0 >= total) {
        return;
    }

    var row = linear / params.nm;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_src = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let row_in = params.offset_in +
        i1 * params.stride_idx1 +
        i2 * params.stride_idx2 +
        i3 * params.stride_idx3;

    let row_out = params.offset_out +
        i1 * params.stride_out1 +
        i2 * params.stride_out2 +
        i3 * params.stride_out3;


    var low: u32 = select(0, k0 - len1, k0 > len1);
    var high: u32 = min(k0, len0);

    while (low < high) {
        let mid = (low + high) >> 1;
        let idx0 = idx_in[row_in + start + mid];
        let idx1 = idx_in[row_in + start + params.len + (k0 - mid - 1)];
        if (take_left(idx0, idx1, row_src)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    var i = low;
    var j = k0 - i;
    var k = k0;
    while (k < k1) {
        var take_l = false;
        if (i >= len0) {
            take_l = false;
        } else if (j >= len1) {
            take_l = true;
        } else {
            let idx0 = idx_in[row_in + start + i];
            let idx1 = idx_in[row_in + start + params.len + j];
            take_l = take_left(idx0, idx1, row_src);
        }

        let out_idx = select(
            idx_in[row_in + start + params.len + j],
            idx_in[row_in + start + i],
            take_l);
        idx_out[row_out + start + k] = out_idx;
        i = select(i, i + 1, take_l);
        j = select(j + 1, j, take_l);
        k += 1;
    }
}
