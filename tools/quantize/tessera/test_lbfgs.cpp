#include "tessera-lbfgs.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

// Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
static float rosenbrock_eval(const float * x, float * grad, int64_t n, void * /*ctx*/) {
    (void)n;
    float a = x[0];
    float b = x[1];
    float f = (1.0f - a) * (1.0f - a) + 100.0f * (b - a * a) * (b - a * a);
    grad[0] = -2.0f * (1.0f - a) + 100.0f * 2.0f * (b - a * a) * (-2.0f * a);
    grad[1] = 100.0f * 2.0f * (b - a * a);
    return f;
}

// f(x) = x^2
static float quadratic_eval(const float * x, float * grad, int64_t n, void * /*ctx*/) {
    (void)n;
    grad[0] = 2.0f * x[0];
    return x[0] * x[0];
}

// project onto [-0.5, 0.5]
static void clip_project(float * x, int64_t n, void * /*ctx*/) {
    for (int64_t i = 0; i < n; i++) {
        if (x[i] < -0.5f) x[i] = -0.5f;
        if (x[i] >  0.5f) x[i] =  0.5f;
    }
}

int main() {
    int failures = 0;

    // Test 1: Rosenbrock via L-BFGS
    {
        float x[2] = { -1.0f, 1.0f };
        ts_lbfgs_params params = { 500, 10, 1e-5f, 1.0f, false };
        float loss = ts_lbfgs_minimize(x, 2, rosenbrock_eval, nullptr, &params);

        float err_x = std::fabs(x[0] - 1.0f);
        float err_y = std::fabs(x[1] - 1.0f);
        if (err_x > 1e-3f || err_y > 1e-3f) {
            fprintf(stderr, "FAIL: Rosenbrock converged to (%.6f, %.6f), loss=%.6e\n",
                    x[0], x[1], loss);
            failures++;
        } else {
            printf("PASS: Rosenbrock -> (%.6f, %.6f), loss=%.6e\n", x[0], x[1], loss);
        }
    }

    // Test 2: PGD with box constraint
    {
        float x[1] = { 2.0f };
        float loss = ts_pgd_minimize(x, 1, quadratic_eval, nullptr,
                                     clip_project, nullptr,
                                     1000, 0.1f, 1e-6f);

        if (std::fabs(x[0]) > 1e-3f) {
            fprintf(stderr, "FAIL: PGD converged to %.6f, loss=%.6e\n", x[0], loss);
            failures++;
        } else {
            printf("PASS: PGD -> %.6f, loss=%.6e\n", x[0], loss);
        }
    }

    if (failures == 0) {
        printf("All tests passed.\n");
    }
    return failures;
}
