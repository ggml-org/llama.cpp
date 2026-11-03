kernel void kernel_dsv4_hc_pre(
        global const char * x,
        ulong offset_x,
        ulong nbx0,
        ulong nbx1,
        ulong nbx2,
        global const char * weights,
        ulong offset_w,
        ulong nbw0,
        ulong nbw1,
        ulong nbw2,
        global char * dst,
        ulong offset_dst,
        long n_embd,
        long n_hc,
        float scale,
        int gated
) {
    const size_t i = get_global_id(0);
    const size_t t = get_global_id(1);
    if (i >= n_embd) {
        return;
    }

    x += offset_x + i*nbx0 + t*nbx2;
    weights += offset_w + (gated ? i*nbw0 + t*nbw2 : t*nbw1);

    float sum = 0.0f;
    for (long h = 0; h < n_hc; ++h) {
        const float xv = *(global const float *) (x + h*nbx1);
        float wv = *(global const float *) (weights + h*(gated ? nbw1 : nbw0));
        if (gated) {
            wv = 1.0f / (1.0f + exp(-wv));
        }
        sum += xv * wv;
    }
    ((global float *) (dst + offset_dst))[i + t*n_embd] = scale * sum;
}

kernel void kernel_dsv4_hc_post(
        global const char * x,
        ulong offset_x,
        ulong nbx0,
        ulong nbx1,
        ulong nbx2,
        global const char * residual,
        ulong offset_r,
        ulong nbr0,
        ulong nbr1,
        ulong nbr2,
        global const char * post,
        ulong offset_p,
        ulong nbp0,
        ulong nbp1,
        ulong nbp2,
        global const char * comb,
        ulong offset_c,
        ulong nbc0,
        ulong nbc1,
        ulong nbc2,
        global char * dst,
        ulong offset_dst,
        long n_embd,
        long n_hc
) {
    const size_t i = get_global_id(0);
    const size_t h = get_global_id(1);
    const size_t t = get_global_id(2);
    if (i >= n_embd) {
        return;
    }

    const float xv = *(global const float *) (x + offset_x + i*nbx0 + t*nbx1);
    const float pv = *(global const float *) (post + offset_p + h*nbp0 + t*nbp1);
    residual += offset_r + i*nbr0 + t*nbr2;

    float sum = xv * pv;
    if (comb) {
        comb += offset_c + h*nbc0 + t*nbc2;
        for (long s = 0; s < n_hc; ++s) {
            const float rv = *(global const float *) (residual + s*nbr1);
            const float cv = *(global const float *) (comb + s*nbc1);
            sum += rv * cv;
        }
    } else {
        sum += *(global const float *) (residual + h*nbr1);
    }
    ((global float *) (dst + offset_dst))[i + (h + t*n_hc)*n_embd] = sum;
}
