#define(VARIANTS)

[
  {
    "SHADER_SUFFIX": "f16_vec",
    "REPLS": {
      "TYPE" : "vec4<f32>",
      "DST_TYPE": "vec4<f16>",
      "VEC_SIZE": 4
    }
  },
  {
    "SHADER_SUFFIX": "f16",
    "REPLS": {
      "TYPE" : "f32",
      "DST_TYPE": "f16",
      "VEC_SIZE": 1
    }
  }
]

#end(VARIANTS)

#define(SHADER)

enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<{{DST_TYPE}}>;

@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(4)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= (params.ne3 * params.ne2 * params.n_rows * params.ne0) / {{VEC_SIZE}}) {
        return;
    }

    // getting the row from gid
    let elems_per_row = params.ne0 / {{VEC_SIZE}};
    var i = gid.x / elems_per_row;

    let i_src3 = i / (params.ne2 * params.n_rows);

    i = i % (params.ne2 * params.n_rows);
    let i_src2 = i / params.n_rows;
    let i_src1 = i % params.n_rows;

    let i_idx2 = i_src3 % params.idx2;
    let i_idx1 = i_src2 % params.idx1;
    let i_idx0 = i_src1;

    let idx_high = (params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2) * 2;

    let idx_high_val = idx[idx_high];
    let idx_low_val = idx[idx_high + 1];

    if (idx_low_val != 0) {
        // Upper bits of index are not zero, output will be incorrect
        atomicStore(&error, 1);
        return;
    }

    let i_dst_row = params.offset_dst + idx_high_val * params.stride_dst1 + i_src2 * params.stride_dst2 + i_src3 * params.stride_dst3;
    let i_src_row = params.offset_src + i_src1 * params.stride_src1 + i_src2 * params.stride_src2 + i_src3 * params.stride_src3;

    let col_idx = (gid.x % elems_per_row);
    dst[i_dst_row/{{VEC_SIZE}} + col_idx] = {{DST_TYPE}}(src[i_src_row/{{VEC_SIZE}} + col_idx]);
}

#end(SHADER)

