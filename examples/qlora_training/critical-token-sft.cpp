#include "critical-token-sft.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <nlohmann/json.hpp>

static std::string json_value_preview(const nlohmann::json & value) {
    std::string result = value.dump();
    if (result.size() > 120) {
        result.resize(117);
        result += "...";
    }
    return result;
}

bool critical_token_parse_spans(
        const nlohmann::json       & record,
        const std::string          & response,
        float                        default_weight,
        std::vector<critical_span> & spans,
        std::string                & error) {
    spans.clear();
    error.clear();
    if (!record.contains("critical_spans")) {
        return true;
    }
    const nlohmann::json & value = record["critical_spans"];
    if (!value.is_array()) {
        error = "critical_spans must be an array, got " + json_value_preview(value);
        return false;
    }

    spans.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const nlohmann::json & item = value[i];
        const std::string field = "critical_spans[" + std::to_string(i) + "]";
        if (!item.is_object()) {
            error = field + " must be an object with integer start/end and optional numeric weight, got " + json_value_preview(item);
            return false;
        }
        if (!item.contains("start") || !item.contains("end") ||
            !(item["start"].is_number_integer() || item["start"].is_number_unsigned()) ||
            !(item["end"].is_number_integer() || item["end"].is_number_unsigned())) {
            error = field + " requires integer start and end, got " + json_value_preview(item);
            return false;
        }

        int64_t start = -1;
        int64_t end = -1;
        try {
            start = item["start"].get<int64_t>();
            end = item["end"].get<int64_t>();
        } catch (const std::exception &) {
            error = field + " start/end are outside the supported integer range, got " + json_value_preview(item);
            return false;
        }

        double weight = default_weight;
        if (item.contains("weight")) {
            if (!item["weight"].is_number()) {
                error = field + ".weight must be numeric, got " + json_value_preview(item["weight"]);
                return false;
            }
            try {
                weight = item["weight"].get<double>();
            } catch (const std::exception &) {
                error = field + ".weight is outside the supported numeric range, got " + json_value_preview(item["weight"]);
                return false;
            }
        }
        if (start < 0 || end <= start || (uint64_t) end > response.size()) {
            error = field + " must satisfy 0 <= start < end <= response byte length " + std::to_string(response.size()) + ", got " + json_value_preview(item);
            return false;
        }
        if (!std::isfinite(weight) || weight < 1.0 || weight > std::numeric_limits<float>::max()) {
            error = field + ".weight must be finite and at least 1.0, got " + json_value_preview(item.contains("weight") ? item["weight"] : nlohmann::json(default_weight));
            return false;
        }
        spans.push_back({ (size_t) start, (size_t) end, (float) weight });
    }
    return true;
}

bool critical_token_apply_spans(
        size_t                                    response_size,
        const std::vector<critical_span>        & spans,
        const std::vector<critical_token_range> & token_ranges,
        std::vector<float>                      & token_weights,
        std::string                             & error) {
    token_weights.assign(token_ranges.size(), 1.0f);

    for (size_t i = 0; i < spans.size(); ++i) {
        const critical_span & span = spans[i];
        if (span.start >= span.end || span.end > response_size) {
            error = "critical_spans[" + std::to_string(i) + "] must satisfy 0 <= start < end <= response byte length " + std::to_string(response_size);
            return false;
        }
        if (!std::isfinite(span.weight) || span.weight < 1.0f) {
            error = "critical_spans[" + std::to_string(i) + "].weight must be finite and at least 1.0";
            return false;
        }
    }

    for (size_t i = 0; i < token_ranges.size(); ++i) {
        const critical_token_range & token = token_ranges[i];
        if (token.start > token.end || token.end > response_size) {
            error = "token byte range is outside the response";
            return false;
        }
        for (const critical_span & span : spans) {
            if (token.start < span.end && span.start < token.end) {
                token_weights[i] = std::max(token_weights[i], span.weight);
            }
        }
    }

    return true;
}

double critical_token_weighted_mean(
        const std::vector<float> & losses,
        const std::vector<float> & weights,
        const std::vector<bool>  & active) {
    if (losses.size() != weights.size() || losses.size() != active.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double numerator = 0.0;
    double denominator = 0.0;
    for (size_t i = 0; i < losses.size(); ++i) {
        if (!active[i]) {
            continue;
        }
        numerator += (double) losses[i] * weights[i];
        denominator += weights[i];
    }
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

std::vector<size_t> critical_token_select_lowest_confidence(
        const std::vector<float> & probabilities,
        const std::vector<bool>  & active,
        float                      threshold,
        float                      max_fraction) {
    std::vector<std::pair<float, size_t>> candidates;
    size_t n_active = 0;
    for (size_t i = 0; i < probabilities.size() && i < active.size(); ++i) {
        if (!active[i]) {
            continue;
        }
        ++n_active;
        if (probabilities[i] < threshold) {
            candidates.push_back({ probabilities[i], i });
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });

    const size_t limit = (size_t) std::floor((double) n_active * max_fraction);
    if (candidates.size() > limit) {
        candidates.resize(limit);
    }

    std::vector<size_t> result;
    result.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        result.push_back(candidate.second);
    }
    return result;
}

float critical_token_confidence_weight(
        float probability,
        float threshold,
        float critical_weight,
        bool  linear) {
    if (probability >= threshold) {
        return 1.0f;
    }
    if (!linear) {
        return critical_weight;
    }
    const float result = 1.0f + (critical_weight - 1.0f) * (1.0f - probability / threshold);
    return std::max(1.0f, std::min(critical_weight, result));
}

float critical_token_warmup_scale(int64_t step, int32_t warmup_steps) {
    if (warmup_steps == 0) {
        return 1.0f;
    }
    return std::min(1.0f, (float) step / warmup_steps);
}
