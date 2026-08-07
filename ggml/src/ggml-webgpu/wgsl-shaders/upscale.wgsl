#if defined(SRC_F16) || defined(DST_F16)
enable f16;
#endif

#ifdef SRC_F16
#define SRC_TYPE f16
#else
#define SRC_TYPE f32
#endif

#ifdef DST_F16
#define DST_TYPE f16
#else
#define DST_TYPE f32
#endif

@group(0) @binding(0)
var<storage, read_write> input: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read_write> output: array<DST_TYPE>;

struct Params {
    offset_i: u32,
    offset_o: u32,

    // element strides
    si0: u32, si1: u32, si2: u32, si3: u32,
    so0: u32, so1: u32, so2: u32, so3: u32,

    src_w: u32,
    src_h: u32,
    src_z: u32,
    src_n: u32,

    dst_w: u32,
    dst_h: u32,
    dst_z: u32,
    dst_n: u32,

    mode_flags: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

const GGML_SCALE_FLAG_ALIGN_CORNERS: u32 = 1u << 8u;

fn get_clamped_input(x: i32, y: i32, z: u32, n: u32) -> f32 {
    let cx = u32(clamp(x, 0, i32(params.src_w) - 1));
    let cy = u32(clamp(y, 0, i32(params.src_h) - 1));
    let i = params.offset_i + cx * params.si0 + cy * params.si1 + z * params.si2 + n * params.si3;
    return f32(input[i]);
}

fn cubic_weight(t: f32, a: f32) -> f32 {
    let at = abs(t);
    if (at <= 1.0) {
        return (a + 2.0) * at * at * at - (a + 3.0) * at * at + 1.0;
    } else if (at <= 2.0) {
        return a * at * at * at - 5.0 * a * at * at + 8.0 * a * at - 4.0 * a;
    } else {
        return 0.0;
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {

    let i_out = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    let total = params.dst_w * params.dst_h * params.dst_z * params.dst_n;

    if (i_out >= total) {
        return;
    }

    // decode (x, y, z, n)
    var i = i_out;
    let x_dst = i % params.dst_w;
    i = i / params.dst_w;
    let y_dst = i % params.dst_h;
    i = i / params.dst_h;
    let z_dst = i % params.dst_z;
    let n_dst = i / params.dst_z;

    // scale factors
    var sf0 = f32(params.dst_w) / f32(params.src_w);
    var sf1 = f32(params.dst_h) / f32(params.src_h);
    var sf2 = f32(params.dst_z) / f32(params.src_z);
    var sf3 = f32(params.dst_n) / f32(params.src_n);

    let align_corners = (params.mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) != 0;

    // pixel_offset: 0.5 for half-pixel-center (default), 0.0 for align_corners
    var pixel_offset = 0.5;
    if (align_corners) {
        pixel_offset = 0.0;
        if (params.dst_w > 1 && params.src_w > 1) {
            sf0 = f32(params.dst_w - 1) / f32(params.src_w - 1);
        }
        if (params.dst_h > 1 && params.src_h > 1) {
            sf1 = f32(params.dst_h - 1) / f32(params.src_h - 1);
        }
    }

    let z_src = min(params.src_z - 1, u32(floor(f32(z_dst) / sf2)));
    let n_src = min(params.src_n - 1, u32(floor(f32(n_dst) / sf3)));

    var result = 0.0;

#if defined(NEAREST)

    let x_src = min(params.src_w - 1, u32(floor(f32(x_dst) / sf0)));
    let y_src = min(params.src_h - 1, u32(floor(f32(y_dst) / sf1)));

    result = get_clamped_input(i32(x_src), i32(y_src), z_src, n_src);

#elif defined(BILINEAR)

#if defined(ANTIALIAS)

    // Antialiased bilinear: triangle filter over a variable support region.
    let support0 = max(1.0f / sf0, 1.0f);
    let support1 = max(1.0f / sf1, 1.0f);
    let invscale0 = 1.0 / support0;
    let invscale1 = 1.0 / support1;

    let fx = (f32(x_dst) + pixel_offset) / sf0;
    let fy = (f32(y_dst) + pixel_offset) / sf1;

    let x_min = max(i32(fx - support0 + pixel_offset), 0);
    let y_min = max(i32(fy - support1 + pixel_offset), 0);
    let x_max = min(i32(fx + support0 + pixel_offset), i32(params.src_w));
    let y_max = min(i32(fy + support1 + pixel_offset), i32(params.src_h));

    var weighted_sum = 0.0;
    var total_weight = 0.0;

    for (var x = x_min; x < x_max; x += 1) {
        let wx = max(1.0 - abs(f32(x) - fx + pixel_offset) * invscale0, 0.0);
        for (var y = y_min; y < y_max; y += 1) {
            let wy = max(1.0 - abs(f32(y) - fy + pixel_offset) * invscale1, 0.0);
            let w = wx * wy;
            if (w > 0.0) {
                weighted_sum += get_clamped_input(x, y, z_src, n_src) * w;
                total_weight += w;
            }
        }
    }

    if (total_weight > 0.0) {
        result = weighted_sum / total_weight;
    }

#else

    let fx = (f32(x_dst) + pixel_offset) / sf0 - pixel_offset;
    let fy = (f32(y_dst) + pixel_offset) / sf1 - pixel_offset;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let dx = clamp(fx - f32(x0), 0.0, 1.0);
    let dy = clamp(fy - f32(y0), 0.0, 1.0);
    let a = get_clamped_input(x0, y0, z_src, n_src);
    let b = get_clamped_input(x0 + 1, y0, z_src, n_src);
    let c = get_clamped_input(x0, y0 + 1, z_src, n_src);
    let d = get_clamped_input(x0 + 1, y0 + 1, z_src, n_src);

    let wa = (1.0 - dx) * (1.0 - dy);
    let wb = dx * (1.0 - dy);
    let wc = (1.0 - dx) * dy;
    let wd = dx * dy;

    result = a * wa + b * wb + c * wc + d * wd;

#endif

#elif defined(BICUBIC)

    // bicubic convolution with alpha = -0.75 (PyTorch default)
    let alpha = -0.75;
    let fx = (f32(x_dst) + pixel_offset) / sf0 - pixel_offset;
    let fy = (f32(y_dst) + pixel_offset) / sf1 - pixel_offset;

    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let dx = fx - f32(x0);
    let dy = fy - f32(y0);

    // horizontal weights for offsets -1, 0, 1, 2
    let wx0 = cubic_weight(dx + 1.0, alpha);
    let wx1 = cubic_weight(dx, alpha);
    let wx2 = cubic_weight(1.0 - dx, alpha);
    let wx3 = cubic_weight(2.0 - dx, alpha);

    // vertical weights for offsets -1, 0, 1, 2
    let wy0 = cubic_weight(dy + 1.0, alpha);
    let wy1 = cubic_weight(dy, alpha);
    let wy2 = cubic_weight(1.0 - dy, alpha);
    let wy3 = cubic_weight(2.0 - dy, alpha);

    // intermediate horizontal interpolation for 4x4 grid of pixels
    // x0-1, x0, x0+1, x0+2, y0-1
    let p0 = get_clamped_input(x0 - 1, y0 - 1, z_src, n_src);
    let p1 = get_clamped_input(x0, y0 - 1, z_src, n_src);
    let p2 = get_clamped_input(x0 + 1, y0 - 1, z_src, n_src);
    let p3 = get_clamped_input(x0 + 2, y0 - 1, z_src, n_src);
    let row0 = p0 * wx0 + p1 * wx1 + p2 * wx2 + p3 * wx3;

    // x0-1, x0, x0+1, x0+2, y0
    let q0 = get_clamped_input(x0 - 1, y0, z_src, n_src);
    let q1 = get_clamped_input(x0, y0, z_src, n_src);
    let q2 = get_clamped_input(x0 + 1, y0, z_src, n_src);
    let q3 = get_clamped_input(x0 + 2, y0, z_src, n_src);
    let row1 = q0 * wx0 + q1 * wx1 + q2 * wx2 + q3 * wx3;

    // x0-1, x0, x0+1, x0+2, y0+1
    let r0 = get_clamped_input(x0 - 1, y0 + 1, z_src, n_src);
    let r1 = get_clamped_input(x0, y0 + 1, z_src, n_src);
    let r2 = get_clamped_input(x0 + 1, y0 + 1, z_src, n_src);
    let r3 = get_clamped_input(x0 + 2, y0 + 1, z_src, n_src);
    let row2 = r0 * wx0 + r1 * wx1 + r2 * wx2 + r3 * wx3;

    // x0-1, x0, x0+1, x0+2, y0+2
    let s0 = get_clamped_input(x0 - 1, y0 + 2, z_src, n_src);
    let s1 = get_clamped_input(x0, y0 + 2, z_src, n_src);
    let s2 = get_clamped_input(x0 + 1, y0 + 2, z_src, n_src);
    let s3 = get_clamped_input(x0 + 2, y0 + 2, z_src, n_src);
    let row3 = s0 * wx0 + s1 * wx1 + s2 * wx2 + s3 * wx3;

    // final vertical interpolation
    result = row0 * wy0 + row1 * wy1 + row2 * wy2 + row3 * wy3;

#endif

    let dst_idx = params.offset_o + x_dst * params.so0 + y_dst * params.so1 + z_dst * params.so2 + n_dst * params.so3;
    output[dst_idx] = DST_TYPE(result);
}
