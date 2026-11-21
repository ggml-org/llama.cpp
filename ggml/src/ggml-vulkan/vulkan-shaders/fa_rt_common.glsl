layout (push_constant) uniform parameter {
    uint KV;
    uint HSK;
    uint HSV;
    uint page_size;
    uint pages;
    uint window;
    uint k_stride;
    uint k_head_stride;
    uint v_stride;
    uint v_head_stride;
    uint q_head_stride;
    float scale;
    float error_budget;
    float padding;
    uint exact_bounds;
    uint dense_mode;
    uint light_summary;
    uint block_cols;
    uint cache_summary;
    uint unit_box;
} p;

const float FLOAT_EPS = 1.192092896e-7;
const float FLOAT_MAX = 3.402823466e38;

bool is_finite(float x) {
    return !isnan(x) && !isinf(x);
}

bool is_finite(vec3 x) {
    return !any(isnan(x)) && !any(isinf(x));
}

vec3 key_projection(uint offset) {
    return vec3(float(data_k[offset]), float(data_k[offset + p.HSK / 3]), float(data_k[offset + 2 * p.HSK / 3]));
}

uint summary_offset(uint head, uint page) {
    return (head * p.pages + page) * (p.light_summary != 0 ? 8 : 2 * p.HSK + 8);
}

vec3 write_summary(uint base, uint count, float maximum, bool invalid, vec3 center) {
    center = count != 0 ? center / float(count) : vec3(0.0);
    const float magnitude = length(center);
    const vec3 direction = magnitude > 1e-12 && is_finite(magnitude) ? center / magnitude : vec3(0.0, 0.0, 3.0);
    const uint tail = base + (p.light_summary != 0 ? 4 : 4 + 2 * p.HSK);
    data_s[base] = float(count);
    data_s[base + 1] = maximum;
    data_s[base + 2] = magnitude;
    data_s[base + 3] = float(invalid);
    data_s[tail] = direction.x;
    data_s[tail + 1] = direction.y;
    data_s[tail + 2] = direction.z;
    data_s[tail + 3] = 0.0;
    return direction;
}
