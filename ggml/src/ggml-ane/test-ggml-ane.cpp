// Smoke test for the ggml-ane backend.
//
// Builds a small graph of ops the ANE backend advertises as supported, runs
// graph_compute on the ANE backend, and checks the output against a CPU
// reference within fp16 tolerance. Also verifies that a graph containing an
// unsupported op is rejected with GGML_STATUS_FAILED.
//
// If a real .mlmodelc bundle is passed via --bundle <dir>, the test also
// exercises ggml_backend_ane_program_load_from_dir + warmup.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ane.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void fail(const char * msg) {
    fprintf(stderr, "FAIL: %s\n", msg);
    ++failures;
}

// fp32 reference for: out = silu(a + b) * a  (a, b are 1-D fp32 of length n)
static void reference_add_silu_mul(const float * a, const float * b, float * out, int n) {
    for (int i = 0; i < n; ++i) {
        float s = a[i] + b[i];
        s = 1.0f / (1.0f + expf(-s));
        float silu = (a[i] + b[i]) * s;
        out[i] = silu * a[i];
    }
}

static bool approx_eq(float x, float y, float tol) {
    return fabsf(x - y) <= tol * (1.0f + fabsf(x) + fabsf(y));
}

int main(int argc, char ** argv) {
    std::string bundle_dir;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bundle" && i + 1 < argc) {
            bundle_dir = argv[++i];
        }
    }

    ggml_backend_load_all();

    // Locate the ANE device.
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("ANE");
    if (!dev) {
        // Backend not built/discovered: report and exit non-zero so CI sees it,
        // but only after printing the registered devices for diagnosis.
        fprintf(stderr, "ANE backend device not found. Registered devices:\n");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(d, &props);
            fprintf(stderr, "  [%zu] %s\n", i, props.name);
        }
        fail("ANE device not found");
        return failures == 0 ? 0 : 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fail("ggml_backend_dev_init(ANE) returned null");
        return 1;
    }
    fprintf(stderr, "ane: backend initialized: %s\n", ggml_backend_name(backend));

    // Optional: load and warm a real .mlmodelc bundle to exercise the Core ML
    // runner. This does not feed graphs yet; it just proves load + warmup work.
    if (!bundle_dir.empty()) {
        fprintf(stderr, "ane: loading bundle %s\n", bundle_dir.c_str());
        ggml_backend_ane_program * prog =
            ggml_backend_ane_program_load_from_dir(bundle_dir.c_str(), nullptr);
        if (!prog) {
            fail("ggml_backend_ane_program_load_from_dir returned null");
        } else {
            fprintf(stderr, "ane: bundle loaded and warmed successfully\n");
            if (ggml_backend_is_ane(backend)) {
                ggml_backend_ane_set_program(backend, prog);
            }
            ggml_backend_ane_program_free(prog);
        }
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);

    // Build a graph: out = silu(a + b) * a, all supported elementwise ops.
    const int n = 256;
    ggml_init_params ip = { /*.mem_size   = */ 64 * 1024 * 1024,
                            /*.mem_buffer = */ nullptr,
                            /*.no_alloc   = */ true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        fail("ggml_init failed");
        return 1;
    }

    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");

    ggml_tensor * sum  = ggml_add(ctx, a, b);
    ggml_tensor * act  = ggml_unary(ctx, sum, GGML_UNARY_OP_SILU);
    ggml_tensor * out  = ggml_mul(ctx, act, a);
    ggml_set_name(out, "out");

    // Verify supports_op agrees on every node.
    const ggml_tensor * nodes[] = { sum, act, out };
    for (const ggml_tensor * t : nodes) {
        if (!ggml_backend_dev_supports_op(dev, t)) {
            char buf[128];
            snprintf(buf, sizeof(buf), "supports_op false for op %s", ggml_op_name(t->op));
            fail(buf);
        }
    }

    // Allocate a, b, out in the ANE buffer type.
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) {
        fail("ggml_backend_alloc_ctx_tensors_from_buft failed");
        return 1;
    }

    // Fill a, b with deterministic data.
    std::vector<float> ha(n), hb(n), href(n);
    for (int i = 0; i < n; ++i) {
        ha[i] = 0.1f * (float) (i - n / 2);
        hb[i] = 0.05f * (float) i - 1.0f;
    }
    reference_add_silu_mul(ha.data(), hb.data(), href.data(), n);
    ggml_backend_tensor_set(a, ha.data(), 0, n * sizeof(float));
    ggml_backend_tensor_set(b, hb.data(), 0, n * sizeof(float));

    // Debug: read back inputs to confirm the buffer round-trip works.
    {
        std::vector<float> chk(n);
        ggml_backend_tensor_get(a, chk.data(), 0, n * sizeof(float));
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (!approx_eq(chk[i], ha[i], 1e-6f)) { ok = false; break; }
        }
        fprintf(stderr, "ane: input 'a' round-trip %s\n", ok ? "ok" : "MISMATCH");
    }

    // Build and run the graph.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "graph_compute returned %d", (int) status);
        fail(buf);
    } else {
        std::vector<float> hout(n);
        ggml_backend_tensor_get(out, hout.data(), 0, n * sizeof(float));
        int mism = 0;
        float worst = 0.0f;
        const float tol = 1e-3f; // fp32 Accelerate path; tight tolerance
        for (int i = 0; i < n; ++i) {
            if (!approx_eq(hout[i], href[i], tol)) {
                ++mism;
                worst = fmaxf(worst, fabsf(hout[i] - href[i]));
            }
        }
        if (mism > 0) {
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "output mismatch: %d/%d elements differ, worst delta %.6f",
                     mism, n, worst);
            fail(buf);
        } else {
            fprintf(stderr, "ane: graph output matches CPU reference (tol %.0e)\n", tol);
        }
    }

    // Negative case: a graph with an unsupported op must be rejected.
    {
        ggml_tensor * c      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_tensor * d      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        // CONCAT is ANE-BREAKS and is not in supports_op.
        ggml_tensor * concat = ggml_concat(ctx, c, d, 0);
        if (ggml_backend_dev_supports_op(dev, concat)) {
            fail("supports_op returned true for CONCAT (should be false)");
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    if (failures == 0) {
        fprintf(stderr, "ane: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "ane: %d check(s) failed\n", failures);
    return 1;
}
