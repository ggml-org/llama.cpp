#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

// Geometry and device limits only; live tensor/lease validation stays outside the cache.
struct ggml_cuda_fattn_plan_key {
    int device=0, cc=0, compiled_arch=0, sms=0, warp=0;
    size_t shared_limit=0;
    int key=0, value=0;
    uint32_t heads=0, kv_heads=0, queries=0, tokens=0, spans=0, family=0;
    bool resume=false, dense_strided=false;

    bool operator==(const ggml_cuda_fattn_plan_key & other) const {
        return device == other.device && cc == other.cc && compiled_arch == other.compiled_arch &&
            sms == other.sms && warp == other.warp && shared_limit == other.shared_limit &&
            key == other.key && value == other.value && heads == other.heads && kv_heads == other.kv_heads &&
            queries == other.queries && tokens == other.tokens && spans == other.spans && family == other.family &&
            resume == other.resume && dense_strided == other.dense_strided;
    }
};

// Fixed host storage avoids unbounded growth as context extents or backend instances change.
template<typename Plan>
class ggml_cuda_fattn_plan_cache {
    struct entry { ggml_cuda_fattn_plan_key key; Plan plan{}; bool valid=false; };
    std::array<entry,8> entries{};
    size_t next=0;

public:
    bool find(const ggml_cuda_fattn_plan_key & key, Plan & output) const {
        for (const auto & item : entries) if (item.valid && item.key == key) {
            output=item.plan;
            return true;
        }
        return false;
    }

    void store(const ggml_cuda_fattn_plan_key & key, const Plan & plan) {
        for (auto & item : entries) if (item.valid && item.key == key) {
            item.plan=plan;
            return;
        }
        entries[next]={key,plan,true};
        next=(next+1)%entries.size();
    }
};
