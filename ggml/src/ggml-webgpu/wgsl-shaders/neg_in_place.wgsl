#define(VARIANTS)

[
  {
    "REPLS": {
      "TYPE" : "f32",
    }
  },
  {
    "REPLS": {
      "TYPE" : "f16",
    }
  }
]

#end(VARIANTS)

#define(SHADER)

enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<{{TYPE}}>;

@group(0) @binding(1)
var<uniform> params: Params;


override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        src[gid.x] = -src[gid.x];
    }

}

#end(SHADER)