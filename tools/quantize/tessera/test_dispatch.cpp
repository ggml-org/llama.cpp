#include "tessera-dispatch.h"

#include <cassert>
#include <cstdio>

int main() {
    ts_dispatch_params params;
    params.input_path        = "test.gguf";
    params.output_path       = "test_out.gguf";
    params.imatrix_path      = "";
    params.policy_path       = "";
    params.policy_out_path   = "";
    params.calib_corpus      = "";
    params.evolve_seed       = 42;
    params.evolve_iters      = 8;
    params.evolve_islands    = 4;
    params.evolve_population = 16;
    params.evolve_only       = false;
    params.calibrate_only    = false;
    params.outlier_frac      = 0.005f;
    params.awq_alpha         = "auto";
    params.awq_clip          = 1.0f;
    params.nthreads          = 1;
    params.verbose           = true;

    ts_dispatch_result result;
    std::string err;

    int rc = ts_dispatch_run(&params, &result, &err);
    assert(rc == 0);
    assert(err.empty());

    // 3 tensors quantized
    assert(result.tensors.size() == 3);
    assert(result.n_tensors_quantized == 3);
    assert(result.n_tensors_skipped == 0);

    // each tensor has non-empty packed data
    for (const auto & t : result.tensors) {
        assert(!t.packed.empty());
        assert(!t.page_scales.empty());
        assert(!t.lane_scales.empty());
        assert(!t.outlier_row_offsets.empty());
        assert(t.out_dim == 4);
        assert(t.in_dim == 640);
        printf("  %s: family=%s mse=%.6f alpha=%.3f packed_bytes=%zu\n",
               t.name.c_str(), t.family.c_str(), t.mse, t.alpha_used,
               t.packed.size());
    }

    // total MSE > 0 (random weights are never perfectly quantized)
    assert(result.total_mse > 0.0f);

    printf("PASS\n");
    return 0;
}
