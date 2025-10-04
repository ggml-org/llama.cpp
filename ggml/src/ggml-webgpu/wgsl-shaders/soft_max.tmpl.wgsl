#define(VARIANTS)
[
  {
    "SHADER_NAME": "soft_max_f32",
    "DECLS": ["BASE_BINDINGS", "NOT_INPLACE", "NO_MASK", "NO_SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_inplace",
    "DECLS": ["BASE_BINDINGS_INPLACE", "INPLACE", "NO_MASK", "NO_SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_sink",
    "DECLS": ["SINK_BINDINGS", "NOT_INPLACE", "NO_MASK", "SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_sink_inplace",
    "DECLS": ["SINK_BINDINGS_INPLACE", "INPLACE", "NO_MASK", "SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f32",
    "REPLS": {
      "MASK_TYPE" : "f32",
    },
    "DECLS": ["MASK_BINDINGS", "NOT_INPLACE", "MASK", "NO_SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f32_inplace",
    "REPLS": {
      "MASK_TYPE" : "f32",
    },
    "DECLS": ["MASK_BINDINGS_INPLACE", "INPLACE", "MASK", "NO_SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f16",
    "REPLS": {
      "MASK_TYPE" : "f16",
    },
    "DECLS": ["MASK_BINDINGS", "NOT_INPLACE", "MASK", "NO_SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f16_inplace",
    "REPLS": {
      "MASK_TYPE" : "f16",
    },
    "DECLS": ["MASK_BINDINGS_INPLACE", "INPLACE", "MASK", "NO_SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f32_sink",
    "REPLS": {
      "MASK_TYPE" : "f32",
    },
    "DECLS": ["MASK_SINK_BINDINGS", "NOT_INPLACE", "MASK", "SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f32_sink_inplace",
    "REPLS": {
      "MASK_TYPE" : "f32",
    },
    "DECLS": ["MASK_SINK_BINDINGS_INPLACE", "INPLACE", "MASK", "SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f16_sink",
    "REPLS": {
      "MASK_TYPE" : "f16",
    },
    "DECLS": ["MASK_SINK_BINDINGS", "NOT_INPLACE", "MASK", "SINK"]
  },
  {
    "SHADER_NAME": "soft_max_f32_mask_f16_sink_inplace",
    "REPLS": {
      "MASK_TYPE" : "f16",
    },
    "DECLS": ["MASK_SINK_BINDINGS_INPLACE", "INPLACE", "MASK", "SINK"]
  }
]
#end(VARIANTS)

#define(DECLS)

#decl(BASE_BINDINGS)
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;
#enddecl(BASE_BINDINGS)

#decl(BASE_BINDINGS_INPLACE)
@group(0) @binding(1)
var<uniform> params: Params;
#enddecl(BASE_BINDINGS_INPLACE)

#decl(SINK_BINDINGS)
@group(0) @binding(1)
var<storage, read_write> sinks: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;
#enddecl(SINK_BINDINGS)

#decl(SINK_BINDINGS_INPLACE)
@group(0) @binding(1)
var<storage, read_write> sinks: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;
#enddecl(SINK_BINDINGS_INPLACE)

#decl(MASK_BINDINGS)
@group(0) @binding(1)
var<storage, read_write> mask: array<{{MASK_TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;
#enddecl(MASK_BINDINGS)

#decl(MASK_BINDINGS_INPLACE)
@group(0) @binding(1)
var<storage, read_write> mask: array<{{MASK_TYPE}}>;

@group(0) @binding(2)
var<uniform> params: Params;
#enddecl(MASK_BINDINGS_INPLACE)

#decl(MASK_SINK_BINDINGS)
@group(0) @binding(1)
var<storage, read_write> mask: array<{{MASK_TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> sinks: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;
#enddecl(MASK_SINK_BINDINGS)

#decl(MASK_SINK_BINDINGS_INPLACE)
@group(0) @binding(1)
var<storage, read_write> mask: array<{{MASK_TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> sinks: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;
#enddecl(MASK_SINK_BINDINGS_INPLACE)

#decl(NOT_INPLACE)
fn inter_value(i: u32) -> f32 {
    return dst[i];
}

fn update(i: u32, val: f32) {
    dst[i] = val;
}
#enddecl(NOT_INPLACE)

#decl(INPLACE)
fn inter_value(i: u32) -> f32 {
    return src[i];
}

fn update(i: u32, val: f32) {
    src[i] = val;
}
#enddecl(INPLACE)

#decl(NO_MASK)
fn mask_val(i: u32) -> f32 {
    return 0.0;
}
#enddecl(NO_MASK)

#decl(MASK)
fn mask_val(i: u32) -> f32 {
    return f32(mask[i]);
}
#enddecl(MASK)

#decl(NO_SINK)
fn lower_max_bound(i2: u32) -> f32 {
    return -1e30;
}

fn add_sinks(val: f32, i2: u32, max_val: f32) -> f32 {
    return val;
}
#enddecl(NO_SINK)

#decl(SINK)
fn lower_max_bound(i2: u32) -> f32 {
    return sinks[params.offset_sinks + i2];
}

fn add_sinks(val: f32, i2: u32, max_val: f32) -> f32 {
    return val + exp(sinks[params.offset_sinks + i2] - max_val);
}
#enddecl(SINK)

#end(DECLS)

#define(SHADER)
enable f16;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_sinks: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // shape of src0/dst
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    // shape of src1
    ne12: u32,
    ne13: u32,

    scale: f32,
    max_bias: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

DECLS

const CACHE_SIZE: u32 = 16;

override wg_size: u32;
var<workgroup> scratch: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src0_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let i_src1_row = params.offset_src1 + (i3 % params.ne13) * params.stride_src13 + (i2 % params.ne12) * params.stride_src12 + i1 * params.stride_src11;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
    let elems = (params.ne0 + wg_size - 1) / wg_size;

    let head = f32(i2);
    let slope = select(1, select(pow(params.m1, 2 * (head - params.n_head_log2) + 1), pow(params.m0, head + 1), head < params.n_head_log2), params.max_bias > 0);

    var cache: array<f32, CACHE_SIZE>;

    var max_val = lower_max_bound(i2);
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        let val = src[i_src0_row + col] * params.scale + slope * mask_val(i_src1_row + col);
        max_val = max(max_val, val);
        if (col < CACHE_SIZE) {
            cache[col] = val;
        }
        col += wg_size;
    }

    scratch[lid.x] = max_val;
    workgroupBarrier();
    var offset = wg_size / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + offset]);
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let row_max = scratch[0];
    workgroupBarrier();

    var sum = 0.0f;
    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        let val = select(src[i_src0_row + col] * params.scale + slope * mask_val(i_src1_row + col),
                         cache[col], col < CACHE_SIZE);
        let ex = exp(val - row_max);
        sum += ex;
        if (col < CACHE_SIZE) {
            cache[col] = ex;
        } else {
            update(i_dst_row + col, ex);
        }
        col += wg_size;
    }

    scratch[lid.x] = sum;
    workgroupBarrier();
    offset = wg_size / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let row_sum = add_sinks(scratch[0], i2, row_max);

    let sum_recip = 1.0 / row_sum;
    col = lid.x;
    for  (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        update(i_dst_row + col, select(inter_value(i_dst_row + col), cache[col], col < CACHE_SIZE) * sum_recip);
        col += wg_size;
    }
}
#end(SHADER)
