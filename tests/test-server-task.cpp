#include "common.h"
#include "server-task.h"

#undef NDEBUG
#include <cassert>

int main() {
    common_params params_base;
    const std::vector<llama_logit_bias> logit_bias_eog;

    {
        json data = {
            {"speculative.ngram_size_n", 12},
            {"speculative.ngram_size_m", 48},
            {"speculative.ngram_m_hits", 7},
        };

        const auto params = server_task::params_from_json_cmpl(nullptr, params_base, 4096, logit_bias_eog, data);
        assert(params.speculative.ngram_size_n == 12);
        assert(params.speculative.ngram_size_m == 48);
        assert(params.speculative.ngram_min_hits == 7);
    }

    {
        json data = {
            {"speculative.ngram_size_n", 0},
            {"speculative.ngram_size_m", 0},
            {"speculative.ngram_m_hits", 0},
        };

        const auto params = server_task::params_from_json_cmpl(nullptr, params_base, 4096, logit_bias_eog, data);
        assert(params.speculative.ngram_size_n == 1);
        assert(params.speculative.ngram_size_m == 1);
        assert(params.speculative.ngram_min_hits == 1);
    }

    {
        json data = {
            {"speculative.ngram_size_n", 2048},
            {"speculative.ngram_size_m", 2048},
            {"speculative.ngram_m_hits", 2048},
        };

        const auto params = server_task::params_from_json_cmpl(nullptr, params_base, 4096, logit_bias_eog, data);
        assert(params.speculative.ngram_size_n == 1024);
        assert(params.speculative.ngram_size_m == 1024);
        assert(params.speculative.ngram_min_hits == 1024);
    }

    return 0;
}
