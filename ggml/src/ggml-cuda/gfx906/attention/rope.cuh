#pragma once

// GFX906 RoPE kernel using __sincosf() for combined sin/cos computation

#include "../gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(GFX906_ROPE_ENABLED)

#define GFX906_ROPE_BLOCK_SIZE 256

struct gfx906_rope_corr_dims {
    float v[2];
};

struct gfx906_mrope_sections {
    int v[4];
};

template<bool forward>
static __device__ __forceinline__ void gfx906_rope_yarn(
        const float theta_extrap,
        const float freq_scale,
        const gfx906_rope_corr_dims corr_dims,
        const int i0,
        const float ext_factor,
        float mscale,
        float & cos_theta,
        float & sin_theta) {

    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;

    if (ext_factor != 0.0f) {
        const float y = (i0 / 2 - corr_dims.v[0]) / fmaxf(0.001f, corr_dims.v[1] - corr_dims.v[0]);
        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * __logf(1.0f / freq_scale);
    }

    __sincosf(theta, &sin_theta, &cos_theta);
    cos_theta *= mscale;
    sin_theta *= mscale;

    if (!forward) {
        sin_theta = -sin_theta;
    }
}

template<bool forward, bool has_ff, typename T>
static __global__ void gfx906_rope_multi(
        const T * __restrict__ x,
        T * __restrict__ dst,
        const int ne0,
        const int ne1,
        const int ne2,
        const int s1,
        const int s2,
        const int n_dims,
        const int32_t * __restrict__ pos,
        const float freq_scale,
        const float ext_factor,
        const float attn_factor,
        const gfx906_rope_corr_dims corr_dims,
        const float theta_scale,
        const float * __restrict__ freq_factors,
        const gfx906_mrope_sections sections,
        const bool is_imrope) {

    const int i0 = 2 * (blockDim.y * blockIdx.y + threadIdx.y);

    if (i0 >= ne0) return;

    const int row_dst = blockDim.x * blockIdx.x + threadIdx.x;
    const int row_x = row_dst % ne1;
    const int channel_x = row_dst / ne1;

    const int idst = row_dst * ne0 + i0 / 2;
    const int ix = channel_x * s2 + row_x * s1 + i0 / 2;

    if (i0 >= n_dims) {
        dst[idst + i0/2 + 0] = x[ix + i0/2 + 0];
        dst[idst + i0/2 + 1] = x[ix + i0/2 + 1];
        return;
    }

    const float theta_power = __powf(theta_scale, (float)(i0 / 2));

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;

    int pos_idx;
    if (is_imrope) {
        const int mod3 = sector % 3;
        if (mod3 == 1 && sector < 3 * sections.v[1]) {
            pos_idx = channel_x + ne2;
        } else if (mod3 == 2 && sector < 3 * sections.v[2]) {
            pos_idx = channel_x + ne2 * 2;
        } else if (mod3 == 0 && sector < 3 * sections.v[0]) {
            pos_idx = channel_x;
        } else {
            pos_idx = channel_x + ne2 * 3;
        }
    } else {
        if (sector < sections.v[0]) {
            pos_idx = channel_x;
        } else if (sector < sec_w) {
            pos_idx = channel_x + ne2;
        } else if (sector < sec_w + sections.v[2]) {
            pos_idx = channel_x + ne2 * 2;
        } else {
            pos_idx = channel_x + ne2 * 3;
        }
    }

    const float theta_base = pos[pos_idx] * theta_power;
    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta, sin_theta;
    gfx906_rope_yarn<forward>(theta_base / freq_factor, freq_scale, corr_dims,
                               i0, ext_factor, attn_factor, cos_theta, sin_theta);

    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims / 2];

    dst[idst + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template<bool forward, typename T>
static void gfx906_rope_multi_cuda(
        const T * x, T * dst,
        const int ne0, const int ne1, const int ne2,
        const int s1, const int s2, const int n_dims, const int nr,
        const int32_t * pos,
        const float freq_scale, const float freq_base,
        const float ext_factor, const float attn_factor,
        const gfx906_rope_corr_dims corr_dims,
        const float * freq_factors,
        const gfx906_mrope_sections sections,
        const bool is_imrope,
        cudaStream_t stream) {

    GGML_ASSERT(ne0 % 2 == 0);

    const dim3 block_dims(1, GFX906_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2 * GFX906_ROPE_BLOCK_SIZE - 1) / (2 * GFX906_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    if (freq_factors == nullptr) {
        gfx906_rope_multi<forward, false, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    } else {
        gfx906_rope_multi<forward, true, T><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor,
            attn_factor, corr_dims, theta_scale, freq_factors, sections, is_imrope);
    }
}

#endif // GGML_USE_HIP && GFX906_ROPE_ENABLED
