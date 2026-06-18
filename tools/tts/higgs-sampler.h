#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace higgs_audio {

static constexpr int BOC_ID    = 1024;
static constexpr int EOC_ID    = 1025;
static constexpr int STOP_CODE = -1;

struct codebook_layout {
    int num_codebooks = 8;
    int codebook_size = 1026;
    int n_embd        = 2560;

    int n_rows() const {
        return num_codebooks * codebook_size;
    }

    std::size_t n_weights() const {
        return (std::size_t) n_rows() * (std::size_t) n_embd;
    }

    int row_for_code(const int codebook, const int code) const {
        if (codebook < 0 || codebook >= num_codebooks) {
            throw std::out_of_range("codebook index is out of range");
        }
        if (code < 0 || code >= codebook_size) {
            throw std::out_of_range("code is out of range for codebook");
        }

        return codebook * codebook_size + code;
    }
};

struct codebook_sampling_params {
    float temperature = 0.0f;
    int top_k = 0;
};

class delay_sampler {
public:
    explicit delay_sampler(const int num_codebooks)
        : n_codebooks(num_codebooks), last_codes(num_codebooks, STOP_CODE) {
        if (num_codebooks <= 0) {
            throw std::invalid_argument("num_codebooks must be positive");
        }
    }

    std::vector<int> step_from_codes(std::vector<int> codes) {
        if ((int) codes.size() != n_codebooks) {
            throw std::invalid_argument("code count does not match num_codebooks");
        }

        if (generation_done) {
            return std::vector<int>(n_codebooks, STOP_CODE);
        }

        if (delay_count < n_codebooks) {
            const int next_codebook = delay_count + 1;
            for (int i = next_codebook; i < n_codebooks; ++i) {
                codes[i] = BOC_ID;
            }
            ++delay_count;
        } else if (eoc_countdown >= 0) {
            --eoc_countdown;
            if (eoc_countdown <= 0) {
                generation_done = true;
            }
        } else if (codes[0] == EOC_ID) {
            if (n_codebooks <= 2) {
                generation_done = true;
            } else {
                eoc_countdown = n_codebooks - 2;
            }
        }

        if (!generation_done) {
            last_codes = codes;
        }

        return codes;
    }

    std::vector<int> step_greedy(const std::vector<std::vector<float>> & logits) {
        if ((int) logits.size() != n_codebooks) {
            throw std::invalid_argument("logit row count does not match num_codebooks");
        }

        std::vector<int> codes;
        codes.reserve(logits.size());
        for (const auto & row : logits) {
            if (row.empty()) {
                throw std::invalid_argument("logit row must not be empty");
            }

            const auto it = std::max_element(row.begin(), row.end());
            codes.push_back((int) std::distance(row.begin(), it));
        }

        return step_from_codes(std::move(codes));
    }

    int num_codebooks() const {
        return n_codebooks;
    }

    int delay_steps() const {
        return delay_count;
    }

    int pending_eoc_steps() const {
        return eoc_countdown;
    }

    bool done() const {
        return generation_done;
    }

    const std::vector<int> & last() const {
        return last_codes;
    }

private:
    int n_codebooks     = 0;
    int delay_count     = 0;
    int eoc_countdown   = -1;
    bool generation_done = false;
    std::vector<int> last_codes;
};

inline std::vector<float> embed_codebook_frame(
        const codebook_layout & layout,
        const std::vector<float> & embedding_weight,
        const std::vector<int> & codes) {
    if (layout.num_codebooks <= 0 || layout.codebook_size <= 0 || layout.n_embd <= 0) {
        throw std::invalid_argument("invalid codebook layout");
    }
    if (embedding_weight.size() != layout.n_weights()) {
        throw std::invalid_argument("embedding weight size does not match codebook layout");
    }
    if ((int) codes.size() != layout.num_codebooks) {
        throw std::invalid_argument("code count does not match num_codebooks");
    }

    std::vector<float> embd((std::size_t) layout.n_embd, 0.0f);

    for (int cb = 0; cb < layout.num_codebooks; ++cb) {
        const int row = layout.row_for_code(cb, codes[cb]);
        const std::size_t row_offset = (std::size_t) row * (std::size_t) layout.n_embd;
        for (int i = 0; i < layout.n_embd; ++i) {
            embd[(std::size_t) i] += embedding_weight[row_offset + (std::size_t) i];
        }
    }

    return embd;
}

inline std::vector<std::vector<float>> project_codebook_logits(
        const codebook_layout & layout,
        const std::vector<float> & head_weight,
        const std::vector<float> & hidden) {
    if (layout.num_codebooks <= 0 || layout.codebook_size <= 0 || layout.n_embd <= 0) {
        throw std::invalid_argument("invalid codebook layout");
    }
    if (head_weight.size() != layout.n_weights()) {
        throw std::invalid_argument("head weight size does not match codebook layout");
    }
    if ((int) hidden.size() != layout.n_embd) {
        throw std::invalid_argument("hidden size does not match codebook layout");
    }

    std::vector<std::vector<float>> logits(
            (std::size_t) layout.num_codebooks,
            std::vector<float>((std::size_t) layout.codebook_size, 0.0f));

    for (int cb = 0; cb < layout.num_codebooks; ++cb) {
        for (int code = 0; code < layout.codebook_size; ++code) {
            const int row = layout.row_for_code(cb, code);
            const std::size_t row_offset = (std::size_t) row * (std::size_t) layout.n_embd;

            float v = 0.0f;
            for (int i = 0; i < layout.n_embd; ++i) {
                v += head_weight[row_offset + (std::size_t) i] * hidden[(std::size_t) i];
            }
            logits[(std::size_t) cb][(std::size_t) code] = v;
        }
    }

    return logits;
}

inline std::vector<int> greedy_codebook_codes(
        const codebook_layout & layout,
        const std::vector<float> & head_weight,
        const std::vector<float> & hidden) {
    if (layout.num_codebooks <= 0 || layout.codebook_size <= 0 || layout.n_embd <= 0) {
        throw std::invalid_argument("invalid codebook layout");
    }
    if (head_weight.size() != layout.n_weights()) {
        throw std::invalid_argument("head weight size does not match codebook layout");
    }
    if ((int) hidden.size() != layout.n_embd) {
        throw std::invalid_argument("hidden size does not match codebook layout");
    }

    std::vector<int> codes((std::size_t) layout.num_codebooks, 0);

    for (int cb = 0; cb < layout.num_codebooks; ++cb) {
        int best_code = 0;
        float best = 0.0f;
        bool have_best = false;

        for (int code = 0; code < layout.codebook_size; ++code) {
            const int row = layout.row_for_code(cb, code);
            const std::size_t row_offset = (std::size_t) row * (std::size_t) layout.n_embd;

            float v = 0.0f;
            for (int i = 0; i < layout.n_embd; ++i) {
                v += head_weight[row_offset + (std::size_t) i] * hidden[(std::size_t) i];
            }
            if (!have_best || v > best) {
                best = v;
                best_code = code;
                have_best = true;
            }
        }

        codes[(std::size_t) cb] = best_code;
    }

    return codes;
}

inline int sample_codebook_row(
        const float * row,
        const int codebook_size,
        const codebook_sampling_params & params,
        std::mt19937 & rng) {
    if (codebook_size <= 0) {
        throw std::invalid_argument("codebook_size must be positive");
    }

    if (params.temperature <= 0.0f) {
        int best_code = 0;
        float best = row[0];
        for (int code = 1; code < codebook_size; ++code) {
            if (row[code] > best) {
                best = row[code];
                best_code = code;
            }
        }
        return best_code;
    }

    int k = params.top_k <= 0 ? codebook_size : std::min(params.top_k, codebook_size);
    std::vector<int> indices((size_t) codebook_size);
    for (int i = 0; i < codebook_size; ++i) {
        indices[(size_t) i] = i;
    }

    std::partial_sort(
            indices.begin(),
            indices.begin() + k,
            indices.end(),
            [&](const int a, const int b) {
                return row[a] > row[b];
            });
    indices.resize((size_t) k);

    float max_logit = row[indices[0]];
    for (int idx : indices) {
        max_logit = std::max(max_logit, row[idx]);
    }

    std::vector<double> weights((size_t) k, 0.0);
    double sum = 0.0;
    const float inv_temp = 1.0f / params.temperature;
    for (int i = 0; i < k; ++i) {
        const double w = std::exp((double) ((row[indices[(size_t) i]] - max_logit) * inv_temp));
        weights[(size_t) i] = w;
        sum += w;
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return indices[0];
    }

    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return indices[(size_t) dist(rng)];
}

inline std::vector<int> sample_codebook_codes(
        const codebook_layout & layout,
        const std::vector<float> & flat_logits,
        const codebook_sampling_params & params,
        std::mt19937 & rng) {
    if (layout.num_codebooks <= 0 || layout.codebook_size <= 0) {
        throw std::invalid_argument("invalid codebook layout");
    }
    if ((int) flat_logits.size() != layout.num_codebooks * layout.codebook_size) {
        throw std::invalid_argument("flat logit count does not match codebook layout");
    }

    std::vector<int> codes((size_t) layout.num_codebooks);
    for (int cb = 0; cb < layout.num_codebooks; ++cb) {
        const float * row = flat_logits.data() + (size_t) cb * (size_t) layout.codebook_size;
        codes[(size_t) cb] = sample_codebook_row(row, layout.codebook_size, params, rng);
    }
    return codes;
}

inline std::vector<int> sample_codebook_codes(
        const codebook_layout & layout,
        const std::vector<std::vector<float>> & logits,
        const codebook_sampling_params & params,
        std::mt19937 & rng) {
    if ((int) logits.size() != layout.num_codebooks) {
        throw std::invalid_argument("logit row count does not match num_codebooks");
    }

    std::vector<int> codes((size_t) layout.num_codebooks);
    for (int cb = 0; cb < layout.num_codebooks; ++cb) {
        if ((int) logits[(size_t) cb].size() != layout.codebook_size) {
            throw std::invalid_argument("logit row width does not match codebook_size");
        }
        codes[(size_t) cb] = sample_codebook_row(logits[(size_t) cb].data(), layout.codebook_size, params, rng);
    }
    return codes;
}

inline std::vector<int> step_greedy_from_hidden(
        delay_sampler & sampler,
        const codebook_layout & layout,
        const std::vector<float> & head_weight,
        const std::vector<float> & hidden) {
    return sampler.step_from_codes(greedy_codebook_codes(layout, head_weight, hidden));
}

inline std::vector<std::vector<int>> apply_delay_pattern(const std::vector<std::vector<int>> & codes, const int num_codebooks) {
    if (num_codebooks <= 0) {
        throw std::invalid_argument("num_codebooks must be positive");
    }

    if (codes.empty()) {
        return {};
    }

    for (const auto & row : codes) {
        if ((int) row.size() != num_codebooks) {
            throw std::invalid_argument("code row width does not match num_codebooks");
        }
    }

    const std::size_t delayed_len = codes.size() + (std::size_t) num_codebooks - 1;
    std::vector<std::vector<int>> delayed(delayed_len, std::vector<int>(num_codebooks, EOC_ID));

    for (int cb = 0; cb < num_codebooks; ++cb) {
        for (int t = 0; t < cb; ++t) {
            delayed[(std::size_t) t][cb] = BOC_ID;
        }
    }

    for (std::size_t t = 0; t < codes.size(); ++t) {
        for (int cb = 0; cb < num_codebooks; ++cb) {
            delayed[t + (std::size_t) cb][cb] = codes[t][cb];
        }
    }

    return delayed;
}

inline std::vector<std::vector<int>> reverse_delay_pattern(const std::vector<std::vector<int>> & delayed, const int num_codebooks) {
    if (num_codebooks <= 0) {
        throw std::invalid_argument("num_codebooks must be positive");
    }

    if (delayed.empty()) {
        return {};
    }

    if (delayed.size() < (std::size_t) num_codebooks) {
        throw std::invalid_argument("delayed code length is shorter than num_codebooks");
    }

    for (const auto & row : delayed) {
        if ((int) row.size() != num_codebooks) {
            throw std::invalid_argument("delayed row width does not match num_codebooks");
        }
    }

    const std::size_t code_len = delayed.size() - (std::size_t) num_codebooks + 1;
    std::vector<std::vector<int>> codes(code_len, std::vector<int>(num_codebooks));

    for (std::size_t t = 0; t < code_len; ++t) {
        for (int cb = 0; cb < num_codebooks; ++cb) {
            codes[t][cb] = delayed[t + (std::size_t) cb][cb];
        }
    }

    return codes;
}

} // namespace higgs_audio
