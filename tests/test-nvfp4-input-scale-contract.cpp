#include <cmath>
#include <cstdio>

static bool approx_eq(float a, float b, float tol = 1e-7f) {
    return std::fabs(a - b) <= tol;
}

static float runtime_input_scale_from_file_scale(float file_input_scale) {
    if (!std::isfinite(file_input_scale) || file_input_scale <= 0.0f) {
        return 1.0f;
    }

    const float runtime_input_scale = 1.0f / file_input_scale;
    if (!std::isfinite(runtime_input_scale) || runtime_input_scale <= 0.0f) {
        return 1.0f;
    }

    return runtime_input_scale;
}

int main() {
    struct test_case {
        float file_scale;
        float runtime_scale;
    };

    // Base NVFP4 contract:
    // GGUF stores the file-domain activation scale.
    // Runtime CUDA paths consume its reciprocal as the activation pre-scale.
    const test_case cases[] = {
        { 0.125f, 8.0f   },
        { 0.25f,  4.0f   },
        { 0.5f,   2.0f   },
        { 1.0f,   1.0f },
        { 2.0f,   0.5f },
        { 8.0f,   0.125f },
    };

    for (const test_case & tc : cases) {
        const float got = runtime_input_scale_from_file_scale(tc.file_scale);
        if (!approx_eq(got, tc.runtime_scale)) {
            std::fprintf(stderr,
                    "runtime input scale mismatch: file_scale=%.8g expected=%.8g got=%.8g\n",
                    tc.file_scale, tc.runtime_scale, got);
            return 1;
        }
    }

    const float invalids[] = { 0.0f, -1.0f, NAN, INFINITY };
    for (float file_scale : invalids) {
        const float got = runtime_input_scale_from_file_scale(file_scale);
        if (!approx_eq(got, 1.0f)) {
            std::fprintf(stderr,
                    "invalid file scale should map to neutral runtime scale: file_scale=%.8g got=%.8g\n",
                    file_scale, got);
            return 1;
        }
    }

    std::printf("NVFP4 input_scale contract OK\n");
    return 0;
}
