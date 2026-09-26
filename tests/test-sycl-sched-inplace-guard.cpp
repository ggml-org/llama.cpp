// The scheduler must refuse, at allocation time, a graph whose in-place op was
// assigned to a backend that cannot address the tensor it writes into. Before
// this check the op ran on the fallback backend and dereferenced device memory.
//
//   weights on SYCL0 pull a CPY (f32 -> q8_0) onto SYCL0. A SET_ROWS into that
//   q8_0 result with an f16 source is declined by SYCL0 and lands on the CPU,
//   which cannot reach the device buffer: alloc_graph() must return false.
//   Control: the same graph with an f32 source stays on SYCL0, allocates and
//   computes.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <array>
#include <cstdio>
#include <vector>

namespace {

struct case_result {
    bool        alloc_ok = false;
    ggml_status status   = GGML_STATUS_FAILED;
};

static case_result run_case(ggml_backend_t sycl, ggml_backend_t cpu, ggml_type src_type) {
    case_result out;

    const int64_t n_cols = 32;
    const int64_t n_rows = 4;
    const int64_t n_set  = 2;

    ggml_init_params wparams = {
        /* .mem_size   = */ ggml_tensor_overhead() * 2,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * wctx = ggml_init(wparams);
    ggml_tensor * w = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_cols, n_rows);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, sycl);
    if (wbuf == nullptr) {
        std::fprintf(stderr, "failed to allocate the weight on SYCL0\n");
        ggml_free(wctx);
        return out;
    }
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> wdata(n_cols * n_rows);
    for (size_t i = 0; i < wdata.size(); ++i) {
        wdata[i] = 0.25f * (float) (i % 7) - 0.5f;
    }
    ggml_backend_tensor_set(w, wdata.data(), 0, wdata.size() * sizeof(float));

    ggml_init_params gparams = {
        /* .mem_size   = */ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(gparams);

    ggml_tensor * q8 = ggml_cast(ctx, w, GGML_TYPE_Q8_0);
    ggml_set_name(q8, "q8");
    ggml_tensor * src = ggml_new_tensor_2d(ctx, src_type, n_cols, n_set);
    ggml_set_name(src, "src");
    ggml_set_input(src);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_set);
    ggml_set_name(idx, "idx");
    ggml_set_input(idx);
    ggml_tensor * set = ggml_set_rows(ctx, q8, src, idx);
    ggml_set_name(set, "set");
    ggml_set_output(set);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, set);

    std::array<ggml_backend_t, 2> backends = { sycl, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), 64, false, true);

    out.alloc_ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (out.alloc_ok) {
        std::vector<float>   src_f32(n_cols * n_set, 1.0f);
        std::vector<int64_t> idx_data = { 1, 3 };
        if (src_type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(src, src_f32.data(), 0, src_f32.size() * sizeof(float));
        } else {
            std::vector<ggml_fp16_t> src_f16(src_f32.size());
            ggml_fp32_to_fp16_row(src_f32.data(), src_f16.data(), (int64_t) src_f32.size());
            ggml_backend_tensor_set(src, src_f16.data(), 0, src_f16.size() * sizeof(ggml_fp16_t));
        }
        ggml_backend_tensor_set(idx, idx_data.data(), 0, idx_data.size() * sizeof(int64_t));
        out.status = ggml_backend_sched_graph_compute(sched, gf);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return out;
}

} // namespace

int main() {
    ggml_backend_t sycl = ggml_backend_sycl_init(0);
    if (sycl == nullptr) {
        std::fprintf(stderr, "failed to initialize SYCL backend 0\n");
        return 1;
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        std::fprintf(stderr, "failed to initialize the CPU backend\n");
        ggml_backend_free(sycl);
        return 1;
    }

    int rc = 0;

    const case_result control = run_case(sycl, cpu, GGML_TYPE_F32);
    if (!control.alloc_ok || control.status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "control (f32 source): expected alloc + compute on SYCL0, got alloc=%d status=%s\n",
                     control.alloc_ok, ggml_status_to_string(control.status));
        rc = 1;
    }

    const case_result guarded = run_case(sycl, cpu, GGML_TYPE_F16);
    if (guarded.alloc_ok) {
        std::fprintf(stderr, "guarded (f16 source): expected alloc_graph() to refuse the CPU fallback, got alloc=1 status=%s\n",
                     ggml_status_to_string(guarded.status));
        rc = 1;
    }

    if (rc == 0) {
        std::fprintf(stderr, "main : in-place scatter with an unreachable destination refused at allocation\n");
    }

    ggml_backend_free(cpu);
    ggml_backend_free(sycl);
    return rc;
}
