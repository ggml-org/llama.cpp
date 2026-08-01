#pragma once

// ts_sharded_map: a simple sharded concurrent hash map.
//
// Splits keys across N shards (default 16), each with its own mutex. Two
// threads touching different keys hash to (likely) different shards and do
// not contend. Within a shard, operations are serialized by that shard's
// mutex. The `with_lock` helper runs a lambda atomically under the shard
// lock, so find-then-update sequences are race-free per shard.
//
// This is the right structure for the GA eval_ctx: layers are disjoint
// across worker threads (the work queue assigns distinct layer indices), so
// each key is written by exactly one thread. The sharding only needs to
// protect the hash table structure against concurrent inserts on different
// keys (which would race on rehash in a plain unordered_map).
//
// Header-only, no dependencies beyond the standard library.

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>

template <typename K, typename V, size_t N = 16>
class ts_sharded_map {
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of two");

    struct shard {
        mutable std::mutex mu;
        std::unordered_map<K, V> map;
    };

    std::array<shard, N> shards_;

    inline shard & shard_for(const K & key) {
        size_t h = std::hash<K>{}(key);
        return shards_[h & (N - 1)];
    }
    inline const shard & shard_for(const K & key) const {
        size_t h = std::hash<K>{}(key);
        return shards_[h & (N - 1)];
    }

public:
    // Run fn under the shard lock for `key`. fn receives a reference to the
    // shard's unordered_map and the key, and may do any combination of
    // find/insert/update atomically. Returns whatever fn returns.
    //
    // Example (lazy load):
    //   map.with_lock(name, [&](auto & m, const auto & k) -> V * {
    //       auto it = m.find(k);
    //       if (it == m.end()) {
    //           V v = load_from_disk(k);
    //           it = m.emplace(k, std::move(v)).first;
    //       }
    //       return &it->second;
    //   });
    template <typename Fn>
    auto with_lock(const K & key, Fn fn)
        -> decltype(fn(std::declval<std::unordered_map<K, V> &>(), key))
    {
        shard & s = shard_for(key);
        std::lock_guard<std::mutex> lk(s.mu);
        return fn(s.map, key);
    }

    // Const variant for read-only access (still locks the shard).
    template <typename Fn>
    auto with_lock(const K & key, Fn fn) const
        -> decltype(fn(std::declval<const std::unordered_map<K, V> &>(), key))
    {
        const shard & s = shard_for(key);
        std::lock_guard<std::mutex> lk(s.mu);
        return fn(s.map, key);
    }

    // Convenience: find a copy of the value for a key (locks, copies, unlocks).
    // Returns true if found. For large values (vectors), use with_lock instead
    // to avoid the copy.
    bool find_copy(const K & key, V & out) const {
        return with_lock(key, [&](const std::unordered_map<K, V> & m, const K & k) {
            auto it = m.find(k);
            if (it == m.end()) return false;
            out = it->second;
            return true;
        });
    }

    // Insert or assign (upsert).
    void assign(const K & key, V value) {
        with_lock(key, [&](std::unordered_map<K, V> & m, const K & k) {
            auto it = m.find(k);
            if (it == m.end()) {
                m.emplace(k, std::move(value));
            } else {
                it->second = std::move(value);
            }
        });
    }

    // Pre-reserve buckets in every shard. Call after construction if the
    // approximate number of keys is known, to avoid rehashing during the GA.
    void reserve_all(size_t n_per_shard) {
        for (auto & s : shards_) {
            std::lock_guard<std::mutex> lk(s.mu);
            s.map.reserve(n_per_shard);
        }
    }

    size_t size() const {
        size_t total = 0;
        for (const auto & s : shards_) {
            std::lock_guard<std::mutex> lk(s.mu);
            total += s.map.size();
        }
        return total;
    }
};
