#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-paged-scheduler-impl.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#define TEST(name) static void name()
#define RUN(name)                                   \
    do {                                            \
        fprintf(stderr, "  running %-40s ", #name); \
        fflush(stderr);                             \
        name();                                     \
        fprintf(stderr, "OK\n");                    \
    } while (0)

#define EXPECT_EQ(a, b)                                                                                          \
    do {                                                                                                         \
        auto _a = (a);                                                                                           \
        auto _b = (b);                                                                                           \
        if (_a != _b) {                                                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: expected %s == %s, got %lld vs %lld\n", __FILE__, __LINE__, #a, #b, \
                    (long long) _a, (long long) _b);                                                             \
        }                                                                                                        \
    } while (0)

#define EXPECT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

// Testing block_manager main functionality

TEST(test_block_manager_leak_simple) {
    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);

    auto gpu_ids = block_manager.checkout_gpu_blocks(10);
    EXPECT_EQ(gpu_ids.size(), 10u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (n_gpu_blocks - 10u));

    auto cpu_ids = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(cpu_ids.size(), 5u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), (n_cpu_blocks - 5u));

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

// Testing that we always return to full (stress test)
TEST(test_block_manager_leak_repeated) {
    const uint32_t block_size   = 16;
    const uint32_t n_gpu_blocks = 64;
    const uint32_t n_cpu_blocks = 32;
    const float    watermark    = 0.0f;
    const int      n_iter       = 1000;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    for (int iter = 0; iter < n_iter; ++iter) {
        const uint32_t n   = (iter % block_size) + 1;
        auto           ids = block_manager.checkout_gpu_blocks(n);
        EXPECT_EQ(ids.size(), (size_t) n);
        block_manager.release_gpu_blocks(ids);
        EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    }
}

// Attempting to check-out more blocks than available. It should return empty.
TEST(test_block_manager_checkout_too_many) {
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto ids = block_manager.checkout_gpu_blocks(9);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);

    auto ids2 = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(ids2.size(), 0u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

TEST(test_block_manager_watermark) {
    // watermark=0.2 means 2 blocks reserved as safety (always consider max gpu blocks)
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.2f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    // 10 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(9));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(10));

    auto ids = block_manager.checkout_gpu_blocks(5);
    EXPECT_EQ(ids.size(), 5u);
    // 5 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(3));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(4));

    block_manager.release_gpu_blocks(ids);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
}

TEST(test_block_manager_watermark_zero) {
    // watermark=0 means the entire pool is requestable.
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(10));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(11));
}

TEST(test_block_manager_gpu_cpu_disjoint) {
    // Just making sure CPU and GPU blocks are disjoint
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto gpu_ids = block_manager.checkout_gpu_blocks(8);
    auto cpu_ids = block_manager.checkout_cpu_blocks(4);

    for (auto id : gpu_ids) {
        EXPECT_TRUE(block_manager.is_gpu(id));
    }
    for (auto id : cpu_ids) {
        EXPECT_FALSE(block_manager.is_gpu(id));
    }

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);
}

// Testing llama_kv_cache_paged book-keeping

// kv_cache_paged with arbitrary shape (we only care about the bookkeeping)
static llama_kv_cache_paged make_kv() {
    return llama_kv_cache_paged(
        /*head_dim=*/64,
        /*n_heads_kv=*/4,
        /*block_size=*/16,
        /*n_layers=*/2,
        /*n_ubatch=*/32,
        /*n_seq_max=*/8);
}

TEST(test_seq_pos_default_unknown) {
    auto kv = make_kv();
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
    EXPECT_EQ(kv.seq_pos_min(42), -1);
}

TEST(test_seq_pos_set_and_get) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);
    EXPECT_EQ(kv.seq_pos_max(0), 17);

    // Independent per seq_id.
    kv.set_seq_min_pos(1, 100);
    EXPECT_EQ(kv.seq_pos_min(1), 100);
    EXPECT_EQ(kv.seq_pos_min(0), 5);  // unchanged
}

TEST(test_seq_pos_overwrite) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(0, 9);
    EXPECT_EQ(kv.seq_pos_min(0), 9);
}

TEST(test_seq_pos_seq_rm_removes) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);

    kv.seq_rm(0, 0, 0);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
}

TEST(test_seq_pos_clear_removes_all) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(1, 7);
    kv.set_seq_min_pos(2, 9);

    kv.clear(/*data=*/false);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_min(1), -1);
    EXPECT_EQ(kv.seq_pos_min(2), -1);
}

TEST(test_free_blocks_releases_to_pool) {
    // Initialize a KV cache with a real CPU backend (used as both "GPU" and CPU).
    // This is fine for testing
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    auto kv = make_kv();
    kv.init(/*backend_gpu=*/backend,
            /*backend_cpu=*/backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, watermark);

    // allocate() pulls from the GPU block pool. After releasing, the count
    // must return to the initial value.
    const uint32_t n_gpu_initial = kv.get_num_gpu_blocks();
    EXPECT_EQ(n_gpu_initial, n_gpu_blocks);

    llama_sequence_group group;
    group.request_id = 0;
    group.n_prompt   = 32;  // 2 blocks
    group.n_decoded  = 0;

    bool success = kv.allocate(/*num_tokens=*/0, group);
    EXPECT_TRUE(success);
    EXPECT_EQ(group.block_table.size(), 2u);

    // Allocating a second sequence further reduces the pool.
    llama_sequence_group group2;
    group2.request_id = 1;
    group2.n_prompt   = 48;  // 3 blocks
    group2.n_decoded  = 0;

    success = kv.allocate(0, group2);
    EXPECT_TRUE(success);
    EXPECT_EQ(group2.block_table.size(), 3u);

    // Free the first sequence: we release 2 blocks
    kv.free_blocks(group);
    EXPECT_EQ(group.block_table.size(), 0u);

    llama_sequence_group group3;
    group3.request_id = 2;
    group3.n_prompt   = 32;  // 2 blocks
    group3.n_decoded  = 0;
    success           = kv.allocate(0, group3);
    EXPECT_TRUE(success);
    EXPECT_EQ(group3.block_table.size(), 2u);

    // All blocks released
    kv.free_blocks(group2);
    kv.free_blocks(group3);

    llama_sequence_group group_full;
    group_full.request_id = 99;
    group_full.n_prompt   = n_gpu_blocks * 16;  // exactly n_gpu_blocks worth
    group_full.n_decoded  = 0;
    success               = kv.allocate(0, group_full);
    EXPECT_TRUE(success);
    EXPECT_EQ(group_full.block_table.size(), 16u);
    kv.free_blocks(group_full);

    ggml_backend_free(backend);
}

// Testing scheduler

// For easy testing and clean-up.
// KV cache paged + scheduler (must free CPU backend)
struct paged_test_fixture {
    ggml_backend_t                              backend = nullptr;
    std::unique_ptr<llama_kv_cache_paged>       kv;
    std::unique_ptr<llama_paged_scheduler_impl> sched;

    ~paged_test_fixture() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    // rule of 5
    paged_test_fixture()                                       = default;
    paged_test_fixture(paged_test_fixture &&)                  = default;
    paged_test_fixture & operator=(paged_test_fixture &&)      = default;
    paged_test_fixture(const paged_test_fixture &)             = delete;
    paged_test_fixture & operator=(const paged_test_fixture &) = delete;
};

static paged_test_fixture make_fixture(uint32_t n_ctx        = 128,
                                       uint32_t block_size   = 16,
                                       uint32_t n_batch      = 64,
                                       uint32_t n_gpu_blocks = 4,
                                       uint32_t n_cpu_blocks = 2) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        /*block_size=*/block_size,
        /*n_layers=*/2u,
        /*n_ubatch=*/n_batch,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(n_ctx, block_size, n_batch, fixture.kv.get()));
    return fixture;
}

static llama_sequence_group make_group(int32_t request_id, uint32_t n_prompt) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = request_id;  // control ordering based on request_id
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

TEST(test_scheduler_no_deadlock_on_empty) {
    // No requests queued means no deadlock
    auto                   fixture = make_fixture();
    llama_batch            batch   = {};
    llama_scheduler_status status  = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 0);
}

TEST(test_scheduler_deadlock_oversize_waiting_request) {
    // There are 2 blocks, the waiting request needs 3.
    // Attempt to process prefill for this requets will fail and the request will remain in waiting (deadlock).
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1);
    auto group   = make_group(0, /*n_prompts=*/32);

    bool queued = fixture.sched->queue_request(group);
    EXPECT_TRUE(queued);

    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);

    // Next steps will continue remained deadlocked
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);
}

TEST(test_scheduler_rejects_oversized_prompt) {
    auto fixture = make_fixture(/*n_ctx=*/64, /*block_size=*/16, /*n_batch=*/128,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    bool queued = fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64));
    EXPECT_FALSE(queued);

    queued = fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/128));
    EXPECT_FALSE(queued);

    // A request just under the limit is accepted.
    queued = fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/63));
    EXPECT_TRUE(queued);
}

int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "test-paged-kv: block_manager\n");
    RUN(test_block_manager_leak_simple);
    RUN(test_block_manager_leak_repeated);
    RUN(test_block_manager_checkout_too_many);
    RUN(test_block_manager_watermark);
    RUN(test_block_manager_watermark_zero);
    RUN(test_block_manager_gpu_cpu_disjoint);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged seq_pos\n");
    RUN(test_seq_pos_default_unknown);
    RUN(test_seq_pos_set_and_get);
    RUN(test_seq_pos_overwrite);
    RUN(test_seq_pos_seq_rm_removes);
    RUN(test_seq_pos_clear_removes_all);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged free_blocks\n");
    RUN(test_free_blocks_releases_to_pool);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged scheduler\n");
    RUN(test_scheduler_no_deadlock_on_empty);
    RUN(test_scheduler_deadlock_oversize_waiting_request);
    RUN(test_scheduler_rejects_oversized_prompt);

    fprintf(stderr, "test-paged-kv: ALL PASSED\n");
    return 0;
}
