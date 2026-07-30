#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace tessera {

struct kv_compaction_metadata {
    std::vector<int32_t> positions;
    std::vector<int32_t> sequence_ids;
    std::vector<uint8_t> valid;
    std::vector<int32_t> pending_shifts;

    void validate() const {
        const size_t size = positions.size();
        if (sequence_ids.size() != size || valid.size() != size || pending_shifts.size() != size) {
            throw std::invalid_argument("KV compaction metadata sizes differ");
        }
    }
};

struct kv_compaction_plan {
    std::vector<int32_t> source_to_destination;
    std::vector<float> source_weights;
    std::vector<int32_t> rope_deltas;
    std::vector<int32_t> destination_positions;
    std::vector<int32_t> destination_sequence_ids;
    std::vector<uint8_t> destination_valid;
};

inline bool kv_position_in(int32_t position, int32_t p0, int32_t p1) {
    return position >= std::max(p0, 0) && position < (p1 < 0 ? std::numeric_limits<int32_t>::max() : p1);
}

inline void kv_remove(kv_compaction_metadata & state, int32_t sequence_id, int32_t p0, int32_t p1) {
    state.validate();
    for (size_t index = 0; index < state.valid.size(); ++index) {
        if (state.valid[index] && kv_position_in(state.positions[index], p0, p1) &&
                (sequence_id < 0 || state.sequence_ids[index] == sequence_id)) {
            state.valid[index] = 0;
        }
    }
}

inline void kv_shift(kv_compaction_metadata & state, int32_t sequence_id, int32_t p0, int32_t p1, int32_t delta) {
    state.validate();
    for (size_t index = 0; index < state.valid.size(); ++index) {
        if (state.valid[index] && state.sequence_ids[index] == sequence_id &&
                kv_position_in(state.positions[index], p0, p1)) {
            state.positions[index] += delta;
            state.pending_shifts[index] += delta;
            if (state.positions[index] < 0) {
                state.valid[index] = 0;
            }
        }
    }
}

inline void kv_divide(kv_compaction_metadata & state, int32_t sequence_id, int32_t p0, int32_t p1, int32_t divisor) {
    if (divisor <= 0) {
        throw std::invalid_argument("KV position divisor must be positive");
    }
    state.validate();
    for (size_t index = 0; index < state.valid.size(); ++index) {
        if (state.valid[index] && state.sequence_ids[index] == sequence_id &&
                kv_position_in(state.positions[index], p0, p1)) {
            const int32_t old_position = state.positions[index];
            state.positions[index] /= divisor;
            state.pending_shifts[index] += state.positions[index] - old_position;
        }
    }
}

inline std::vector<size_t> kv_defragment_order(const kv_compaction_metadata & state) {
    state.validate();
    std::vector<size_t> order(state.valid.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_partition(order.begin(), order.end(), [&](size_t index) {
        return state.valid[index] != 0;
    });
    return order;
}

inline kv_compaction_plan kv_make_plan(
        const kv_compaction_metadata & state,
        int32_t target_per_sequence,
        int32_t keep_prefix,
        int32_t keep_recent,
        bool merge,
        const std::vector<float> & importance = {}) {
    state.validate();
    if (target_per_sequence <= 0 || keep_prefix < 0 || keep_recent < 0) {
        throw std::invalid_argument("invalid KV compaction budget");
    }
    if (!importance.empty() && importance.size() != state.valid.size()) {
        throw std::invalid_argument("KV importance size differs from cache");
    }
    const size_t slots = state.valid.size();
    std::vector<std::pair<int32_t, std::vector<size_t>>> groups;
    std::set<int32_t> sequence_ids;
    for (size_t index = 0; index < slots; ++index) {
        if (state.valid[index]) {
            sequence_ids.insert(state.sequence_ids[index]);
        }
    }
    for (const int32_t sequence_id : sequence_ids) {
        std::vector<size_t> indices;
        for (size_t index = 0; index < slots; ++index) {
            if (state.valid[index] && state.sequence_ids[index] == sequence_id) {
                indices.push_back(index);
            }
        }
        std::stable_sort(indices.begin(), indices.end(), [&](size_t left, size_t right) {
            return state.positions[left] < state.positions[right];
        });
        if (indices.size() <= static_cast<size_t>(target_per_sequence)) {
            for (const size_t index : indices) {
                groups.push_back({ sequence_id, { index } });
            }
            continue;
        }
        const int32_t prefix_count = std::min<int32_t>({
            keep_prefix,
            target_per_sequence,
            static_cast<int32_t>(indices.size()),
        });
        const int32_t recent_count = std::min<int32_t>({
            keep_recent,
            target_per_sequence - prefix_count,
            static_cast<int32_t>(indices.size()) - prefix_count,
        });
        const int32_t middle_budget = target_per_sequence - prefix_count - recent_count;
        for (int32_t index = 0; index < prefix_count; ++index) {
            groups.push_back({ sequence_id, { indices[index] } });
        }
        const size_t middle_begin = prefix_count;
        const size_t middle_end = indices.size() - recent_count;
        const size_t middle_size = middle_end - middle_begin;
        if (merge && middle_budget > 0) {
            const size_t base = middle_size / middle_budget;
            const size_t remainder = middle_size % middle_budget;
            size_t cursor = middle_begin;
            for (int32_t group_index = 0; group_index < middle_budget; ++group_index) {
                const size_t count = base + (static_cast<size_t>(group_index) < remainder ? 1 : 0);
                if (count == 0) {
                    continue;
                }
                groups.push_back({
                    sequence_id,
                    std::vector<size_t>(indices.begin() + cursor, indices.begin() + cursor + count),
                });
                cursor += count;
            }
        } else if (!merge && middle_budget > 0) {
            for (size_t index = middle_end - middle_budget; index < middle_end; ++index) {
                groups.push_back({ sequence_id, { indices[index] } });
            }
        }
        for (size_t index = middle_end; index < indices.size(); ++index) {
            groups.push_back({ sequence_id, { indices[index] } });
        }
    }
    if (groups.size() > slots) {
        throw std::invalid_argument("KV compaction plan exceeds physical slots");
    }
    kv_compaction_plan plan = {
        std::vector<int32_t>(slots, -1),
        std::vector<float>(slots, 0.0f),
        std::vector<int32_t>(slots, 0),
        std::vector<int32_t>(slots, -1),
        std::vector<int32_t>(slots, -1),
        std::vector<uint8_t>(slots, 0),
    };
    for (size_t destination = 0; destination < groups.size(); ++destination) {
        const auto & group = groups[destination];
        int32_t destination_position = std::numeric_limits<int32_t>::min();
        float weight_sum = 0.0f;
        bool uniform_weights = importance.empty();
        for (const size_t source : group.second) {
            destination_position = std::max(destination_position, state.positions[source]);
            weight_sum += importance.empty() ? 1.0f : importance[source];
        }
        if (weight_sum == 0.0f) {
            weight_sum = static_cast<float>(group.second.size());
            uniform_weights = true;
        }
        plan.destination_positions[destination] = destination_position;
        plan.destination_sequence_ids[destination] = group.first;
        plan.destination_valid[destination] = 1;
        for (const size_t source : group.second) {
            const float weight = uniform_weights ? 1.0f : importance[source];
            plan.source_to_destination[source] = destination;
            plan.source_weights[source] = weight / weight_sum;
            plan.rope_deltas[source] = destination_position - state.positions[source];
        }
    }
    return plan;
}

}
