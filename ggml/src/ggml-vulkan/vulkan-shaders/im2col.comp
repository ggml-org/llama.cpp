#version 450

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

#include "types.glsl"

layout (push_constant) uniform parameter
{
    BDA_STORAGE_T dst_addr;
    uint batch_offset; uint offset_delta;
    uint IC;
    uint IW; uint IH;
    uint OW; uint OH;
    uint KW; uint KH;
    uint OH_batch;
    uint CHW;
    int s0; int s1;
    int p0; int p1;
    int d0; int d1;
    uint batch_IC;
} p;

layout(constant_id = 0) const uint BLOCK_SIZE = 32;

const uint NUM_ITER = 512 / BLOCK_SIZE;

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

#if BDA
layout (buffer_reference) buffer D_ptr {D_TYPE d;};
#endif

void im2col(const uint ow, const uint z_idx) {
    const uint oh = z_idx % p.OH;
    const uint batch_idx = z_idx / p.OH;

    const uint gidx = gl_LocalInvocationID.x;
    const uint src_batch = batch_idx * p.batch_offset;
    const BDA_OFFSET_T dst_row = ((BDA_OFFSET_T(batch_idx) * p.OH + oh) * p.OW + ow) * p.CHW;

    const uint KHKW = p.KH * p.KW;

    uint wg_x = gl_WorkGroupID.x;
    do {
        const uint wg_offset = wg_x * 512;

        [[unroll]] for (uint i = 0; i < NUM_ITER; ++i) {
            const uint chw_idx = wg_offset + gidx + i * BLOCK_SIZE;

            if (chw_idx >= p.CHW) {
                return;
            }

            const uint ic = chw_idx / KHKW;
            const uint rem = chw_idx - ic * KHKW;
            const uint ky = rem / p.KW;
            const uint kx = rem - ky * p.KW;

            const uint iiw = ow * p.s0 + kx * p.d0 - p.p0;
            const uint iih = oh * p.s1 + ky * p.d1 - p.p1;

            A_TYPE val = A_TYPE(0);
            if (iih < p.IH && iiw < p.IW) {
                val = data_a[src_batch + ic * p.offset_delta + iih * p.IW + iiw];
            }

#if BDA
            D_ptr out_ptr = D_ptr(p.dst_addr + D_SIZE * (dst_row + chw_idx));
            out_ptr.d = D_TYPE(val);
#else
            data_d[dst_row + chw_idx] = D_TYPE(val);
#endif
        }

        wg_x += gl_NumWorkGroups.x;
    } while (wg_x * 512 < p.CHW);
}

void main() {
    uint ow = gl_GlobalInvocationID.y;
    while (ow < p.OW) {
        uint z = gl_GlobalInvocationID.z;
        while (z < p.OH_batch) {
            im2col(ow, z);
            z += gl_NumWorkGroups.z;
        }
        ow += gl_NumWorkGroups.y;
    }
}
