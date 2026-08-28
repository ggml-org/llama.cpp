#pragma once

#include <atomic>
#include <cstdint>

#define SPINE_CACHE_LINE  64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_LINE)))

struct spine_barrier_t {
    SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
    SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
    SPINE_CACHE_ALIGN int64_t              total_;
};

inline void spine_barrier_wait(spine_barrier_t * b) {
    auto cur_round = b->rounds_.load(std::memory_order_acquire);
    auto cnt       = --b->pending_;
    if (cnt == 0) {
        b->pending_.store(b->total_);
        b->rounds_.store(cur_round + 1);
    } else {
        while (cur_round == b->rounds_.load(std::memory_order_relaxed)) {
            __asm__ volatile("pause " ::: "memory");
        }
    }
}

inline void spine_barrier_init(spine_barrier_t * b, int num_barriers, uint64_t thread_count) {
    for (int i = 0; i < num_barriers; i++) {
        b[i].total_ = thread_count;
        b[i].pending_.store(thread_count);
        b[i].rounds_.store(0);
    }
}
