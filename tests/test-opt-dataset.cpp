#include "common.h"

#include <cstdio>

// Regression test for https://github.com/ggml-org/llama.cpp/issues/27648
//
// common_opt_dataset_init() used to compute the number of training data points with
// unsigned size_t arithmetic:
//     const int64_t ndata = (tokens.size() - ne_datapoint - 1) / stride;
// When the training text is shorter than the context size, the subtraction wraps to
// a huge unsigned value (e.g. 400 - 512 - 1 -> 2^64 - 113), and the division yields
// 2^56 - 1 instead of a non-positive count. The dataset tensors were then created
// with an absurd size, and the first training graph allocation crashed with
// "failed to allocate buffer of size 18446744073709547520".
//
// The computation now uses signed arithmetic via common_opt_dataset_ndata().

int main() {
    // text shorter than the context: must yield a non-positive count (was 72057594037927935)
    {
        const int64_t ndata = common_opt_dataset_ndata(/*n_tokens =*/ 400, /*n_ctx =*/ 512, /*stride =*/ 256);
        if (ndata > 0) {
            fprintf(stderr, "test-opt-dataset: FAIL: short text produced ndata = %lld\n", (long long) ndata);
            return 1;
        }
    }

    // text exactly at the boundary n_ctx + 1: no data points
    {
        const int64_t ndata = common_opt_dataset_ndata(513, 512, 256);
        if (ndata != 0) {
            fprintf(stderr, "test-opt-dataset: FAIL: boundary text produced ndata = %lld\n", (long long) ndata);
            return 1;
        }
    }

    // adequate text: exact expected number of data points
    {
        const int64_t ndata = common_opt_dataset_ndata(819, 256, 128);
        if (ndata != 4) {
            fprintf(stderr, "test-opt-dataset: FAIL: expected ndata = 4, got %lld\n", (long long) ndata);
            return 1;
        }
    }

    printf("test-opt-dataset: all tests passed\n");
    return 0;
}
