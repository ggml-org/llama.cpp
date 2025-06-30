kernel void kernel_upscale(
    global const void * p_src0,
    ulong off_src0,
    global void * p_dst,
    ulong off_dst,
    ulong nb00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne10,
    int ne11,
    int ne12,
    int ne13,
    float sf0,
    float sf1,
    float sf2,
    float sf3
) {
    global const char * src_base = (global const char *)p_src0 + off_src0;
    global float * dst_base = (global float *)((global char *)p_dst + off_dst);

    int index = get_global_id(0);
    int dst_total_elements = ne10 * ne11 * ne12 * ne13;

    if (index >= dst_total_elements) {
        return;
    }

    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = index / (ne10 * ne11 * ne12);

    int i00 = (int)(i10 / sf0);
    int i01 = (int)(i11 / sf1);
    int i02 = (int)(i12 / sf2);
    int i03 = (int)(i13 / sf3);

    ulong offset_src_element = (ulong)i03 * nb03 + (ulong)i02 * nb02 + (ulong)i01 * nb01 + (ulong)i00 * nb00;
    global const float * src_element_ptr = (global const float *)(src_base + offset_src_element);

    dst_base[index] = *src_element_ptr;
}

kernel void kernel_upscale_bilinear(
    global const void * p_src0,
    ulong off_src0,
    global void * p_dst,
    ulong off_dst,
    ulong nb00,
    ulong nb01,
    ulong nb02,
    ulong nb03,
    int ne00_src,
    int ne01_src,
    int ne10_dst,
    int ne11_dst,
    int ne12_dst,
    int ne13_dst,
    float sf0,
    float sf1,
    float sf2,
    float sf3
) {
    global const char * src_base = (global const char *)p_src0 + off_src0;
    global float * dst_base = (global float *)((global char *)p_dst + off_dst);

    int index = get_global_id(0);
    int dst_total_elements = ne10_dst * ne11_dst * ne12_dst * ne13_dst;

    if (index >= dst_total_elements) {
        return;
    }

    int i10_dst = index % ne10_dst;
    int i11_dst = (index / ne10_dst) % ne11_dst;
    int i12_dst = (index / (ne10_dst * ne11_dst)) % ne12_dst;
    int i13_dst = index / (ne10_dst * ne11_dst * ne12_dst);

    int i02_src = (int)(i12_dst / sf2);
    int i03_src = (int)(i13_dst / sf3);

    const float pixel_offset = 0.5f;

    float y_src_f = ((float)i11_dst + pixel_offset) / sf1 - pixel_offset;
    long y0_src = (long)floor(y_src_f);
    long y1_src = y0_src + 1;

    y0_src = max(0L, min(y0_src, (long)ne01_src - 1));
    y1_src = max(0L, min(y1_src, (long)ne01_src - 1));

    float dy = y_src_f - (float)y0_src;
    dy = max(0.0f, min(dy, 1.0f));

    float x_src_f = ((float)i10_dst + pixel_offset) / sf0 - pixel_offset;
    long x0_src = (long)floor(x_src_f);
    long x1_src = x0_src + 1;

    x0_src = max(0L, min(x0_src, (long)ne00_src - 1));
    x1_src = max(0L, min(x1_src, (long)ne00_src - 1));

    float dx = x_src_f - (float)x0_src;
    dx = max(0.0f, min(dx, 1.0f));

    global const float * p_a = (global const float *)(src_base + (ulong)x0_src * nb00 + (ulong)y0_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
    global const float * p_b = (global const float *)(src_base + (ulong)x1_src * nb00 + (ulong)y0_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
    global const float * p_c = (global const float *)(src_base + (ulong)x0_src * nb00 + (ulong)y1_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);
    global const float * p_d = (global const float *)(src_base + (ulong)x1_src * nb00 + (ulong)y1_src * nb01 + (ulong)i02_src * nb02 + (ulong)i03_src * nb03);

    const float val_a = *p_a;
    const float val_b = *p_b;
    const float val_c = *p_c;
    const float val_d = *p_d;

    float result = val_a * (1.0f - dx) * (1.0f - dy) +
                   val_b * dx * (1.0f - dy) +
                   val_c * (1.0f - dx) * dy +
                   val_d * dx * dy;

    dst_base[index] = result;
}
