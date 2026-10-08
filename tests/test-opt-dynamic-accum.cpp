// what this catches: ggml-opt with graphs built per step (ggml_opt_prepare_alloc, the path
// llama_context's training takes) must start every optimizer period from ZERO gradient
// accumulators. Each step here is SGD on loss = sum(w * x), so its gradient is x and each
// step moves w by exactly -lr * x. If the accumulators carried the previous period's gradient,
// the second step would move w by -2 lr x, the third by -3 lr x.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); exit(1); } } while (0)

static const float LR = 0.125f;
static const float X  = 2.0f;

static ggml_opt_optimizer_params sgd_pars(void *) {
    ggml_opt_optimizer_params p = ggml_opt_get_default_optimizer_params(nullptr);
    p.sgd.alpha = LR;
    p.sgd.wd    = 0.0f;
    return p;
}

static void run(int32_t opt_period) {
    // by type, not ggml_backend_cpu_init: in a dynamically loaded backend build (CI) the CPU
    // backend is its own library
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);

    ggml_init_params sp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_static = ggml_init(sp);
    ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_param(w);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, cpu);
    float w_now = 1.0f;
    ggml_backend_tensor_set(w, &w_now, 0, sizeof(float));

    ggml_opt_params params = ggml_opt_default_params(sched, GGML_OPT_LOSS_TYPE_SUM);
    params.optimizer    = GGML_OPT_OPTIMIZER_TYPE_SGD;
    params.get_opt_pars = sgd_pars;
    params.opt_period   = opt_period;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params); // no ctx_compute: graphs are built per step
    ggml_opt_result_t  result  = ggml_opt_result_init();

    const int n_evals = 3 * opt_period;
    for (int i = 0; i < n_evals; ++i) {
        ggml_init_params cp = { GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + 4 * ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, true), nullptr, true };
        ggml_context * ctx_compute = ggml_init(cp);
        ggml_tensor * x   = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, 1);
        ggml_tensor * out = ggml_mul(ctx_compute, w, x);
        ggml_cgraph * gf  = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(gf, out);
        ggml_opt_prepare_alloc(opt_ctx, ctx_compute, gf, x, out);
        ggml_opt_alloc(opt_ctx, true);
        ggml_backend_tensor_set(x, &X, 0, sizeof(float));
        ggml_opt_eval(opt_ctx, result);
        ggml_free(ctx_compute);

        float w_after;
        ggml_backend_tensor_get(w, &w_after, 0, sizeof(float));
        if ((i + 1) % opt_period == 0) {
            // one period = opt_period evals of gradient X each, scaled by nothing (LOSS_TYPE_SUM)
            const float expected = w_now - LR * X * opt_period;
            if (std::fabs(w_after - expected) > 1e-6f) {
                fprintf(stderr, "opt_period %d, step %d: w moved to %f, expected %f (from %f)\n",
                        opt_period, (i + 1) / opt_period, w_after, expected, w_now);
                exit(1);
            }
            w_now = w_after;
        } else {
            CHECK(std::fabs(w_after - w_now) < 1e-7f && "no update inside a period");
        }
    }

    ggml_opt_result_free(result);
    ggml_opt_free(opt_ctx);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
}

int main() {
    ggml_backend_load_all();
    run(1);
    run(2);
    printf("test-opt-dynamic-accum: OK\n");
    return 0;
}
