// A DSV4 rollback marker must survive a graph failure that an asynchronous
// backend reports only at the next synchronize, i.e. after the batch context
// that consumed the marker in next() is gone.
//
//   decode prompt, seq_rm(0, p0)        -> marker pending (rollback of 3)
//   decode token p0                     -> enqueued; next() consumes the marker
//   inject a SYCL sync failure          -> llama_synchronize() reports it
//   expected: marker restored and token p0 dropped, so a second partial
//   removal is still refused (single-use marker) and seq_pos_max is p0 - 1;
//   the retry of token p0 then succeeds and consumes the marker for real.
//
// The injection fires in the SYCL backend's synchronize, which the scheduler
// calls for every backend it holds, so it triggers regardless of which ops
// actually ran on the device.
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "llama.h"
#include "../ggml/src/ggml-sycl/ggml-sycl-test.h"

#include <cstdio>
#include <cstring>
#include <vector>

static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tokens, llama_pos p0, llama_pos p1) {
    llama_batch batch = llama_batch_init(p1 - p0, 0, 1);
    for (llama_pos pos = p0; pos < p1; ++pos) {
        const int32_t i = batch.n_tokens++;
        batch.token   [i]    = tokens[pos];
        batch.pos     [i]    = pos;
        batch.n_seq_id[i]    = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i]    = pos + 1 == p1;
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// The flag lives on the device context, so a throwaway backend on the same
// device arms the failure for the backend inside the llama context.
static bool arm_sycl_sync_failure() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_dev_get(i), nullptr);
        if (backend == nullptr) {
            continue;
        }
        const bool armed = ggml_backend_is_sycl(backend) && ggml_backend_sycl_test_inject_sync_failure_once(backend);
        ggml_backend_free(backend);
        if (armed) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        }
    }
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s -m <dsv4 model.gguf>\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    // No offload: the SYCL backend still sits in the scheduler, which is all
    // the injection needs, and the generated DSV4 model does not survive GPU
    // offload on SYCL yet (CPU set_rows fallback asserts on a row index).
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to load %s\n", __func__, model_path);
        return 1;
    }
    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        llama_model_free(model);
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_ctx     = 64;
    cparams.n_batch   = 64;
    cparams.n_ubatch  = 64;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    int rc = 1;
    const auto fail = [&](const char * what) {
        fprintf(stderr, "main : FAIL: %s\n", what);
        llama_free(ctx);
        llama_model_free(model);
        return rc;
    };

    const uint32_t n_rs_seq = llama_n_rs_seq(ctx);
    constexpr uint32_t n_rollback = 3;
    if (n_rs_seq < n_rollback) {
        fprintf(stderr, "%s : skipping because n_rs_seq is too small\n", __func__);
        rc = 0;
        return fail("skip");
    }

    std::vector<llama_token> tokens(n_rs_seq + 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) (1 + i);
    }
    const llama_pos n_tokens = (llama_pos) tokens.size();
    const llama_pos p0       = n_tokens - n_rollback;

    llama_memory_t mem = llama_get_memory(ctx);

    if (!decode_range(ctx, tokens, 0, n_tokens)) {
        return fail("prompt decode");
    }
    llama_synchronize(ctx);
    if (!llama_memory_seq_rm(mem, 0, p0, -1)) {
        return fail("partial removal refused");
    }
    if (llama_memory_seq_rm(mem, 0, p0 - 1, -1)) {
        return fail("second partial removal accepted while a marker is pending");
    }

    // enqueue the replay of p0; on an asynchronous backend decode() returns
    // success and the failure is only reported by the synchronize below
    if (!decode_range(ctx, tokens, p0, p0 + 1)) {
        return fail("replay decode failed synchronously");
    }
    if (!arm_sycl_sync_failure()) {
        fprintf(stderr, "%s : skipping, no SYCL device\n", __func__);
        rc = 0;
        return fail("skip");
    }
    llama_synchronize(ctx);
    if (llama_get_logits_ith(ctx, -1) != nullptr) {
        return fail("injected sync failure was not reported");
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    if (pos_max != p0 - 1) {
        fprintf(stderr, "%s : seq_pos_max %d after the failed replay, expected %d\n", __func__, pos_max, p0 - 1);
        return fail("token of the failed ubatch stayed committed");
    }
    if (llama_memory_seq_rm(mem, 0, p0 - 1, -1)) {
        return fail("rollback marker was consumed by the failed ubatch");
    }

    // the retry must go through and consume the marker for real
    if (!decode_range(ctx, tokens, p0, p0 + 1)) {
        return fail("retry decode");
    }
    llama_synchronize(ctx);
    if (llama_get_logits_ith(ctx, -1) == nullptr) {
        return fail("retry produced no logits");
    }
    if (llama_memory_seq_pos_max(mem, 0) != p0) {
        return fail("retry did not commit token p0");
    }
    if (!llama_memory_seq_rm(mem, 0, p0 - 1, -1)) {
        return fail("marker still pending after the successful retry");
    }

    fprintf(stderr, "%s : rollback marker survived an asynchronously reported failure\n", __func__);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
