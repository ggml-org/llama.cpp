#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

struct critical_span {
    size_t start;
    size_t end;
    float  weight;
};

struct critical_token_range {
    size_t start;
    size_t end;
};

bool critical_token_parse_spans(
        const nlohmann::json       & record,
        const std::string          & response,
        float                        default_weight,
        std::vector<critical_span> & spans,
        std::string                & error);

bool critical_token_apply_spans(
        size_t                                    response_size,
        const std::vector<critical_span>        & spans,
        const std::vector<critical_token_range> & token_ranges,
        std::vector<float>                      & token_weights,
        std::string                             & error);

double critical_token_weighted_mean(
        const std::vector<float> & losses,
        const std::vector<float> & weights,
        const std::vector<bool>  & active);

std::vector<size_t> critical_token_select_lowest_confidence(
        const std::vector<float> & probabilities,
        const std::vector<bool>  & active,
        float                      threshold,
        float                      max_fraction);

float critical_token_confidence_weight(
        float probability,
        float threshold,
        float critical_weight,
        bool  linear);

float critical_token_warmup_scale(int64_t step, int32_t warmup_steps);
