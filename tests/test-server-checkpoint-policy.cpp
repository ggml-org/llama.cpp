#include "../tools/server/server-checkpoint.h"

#include <cstdio>
#include <cstdlib>
#include <list>
#include <sstream>
#include <stdexcept>
#include <vector>

static common_prompt_checkpoint make_checkpoint(int64_t n_tokens, size_t size = 1) {
    common_prompt_checkpoint ckpt;
    ckpt.update_pos(n_tokens, (llama_pos) n_tokens, (llama_pos) n_tokens);
    ckpt.data_tgt.resize(size);
    return ckpt;
}

static std::vector<int64_t> checkpoint_positions(const std::list<common_prompt_checkpoint> & checkpoints) {
    std::vector<int64_t> res;
    for (const auto & ckpt : checkpoints) {
        res.push_back(ckpt.n_tokens);
    }
    return res;
}

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void require_eq(const std::vector<int64_t> & expected, const std::vector<int64_t> & actual) {
    if (expected == actual) {
        return;
    }

    auto format = [](const std::vector<int64_t> & values) {
        std::ostringstream out;
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) {
                out << ", ";
            }
            out << values[i];
        }
        out << "]";
        return out.str();
    };

    throw std::runtime_error("expected " + format(expected) + ", got " + format(actual));
}

static void prune_to(std::list<common_prompt_checkpoint> & checkpoints, size_t max_checkpoints) {
    while (checkpoints.size() > max_checkpoints) {
        checkpoints.erase(server_checkpoint::find_redundant_checkpoint(checkpoints));
    }
}

static void test_ladder_min_step() {
    require(server_checkpoint::ladder_min_step(0, 0) == 0, "zero min step must stay disabled");
    require(server_checkpoint::ladder_min_step(0, 256) == 256, "base ladder step mismatch");
    require(server_checkpoint::ladder_min_step(8191, 256) == 256, "first ladder range mismatch");
    require(server_checkpoint::ladder_min_step(8192, 256) == 512, "second ladder range mismatch");
    require(server_checkpoint::ladder_min_step(16384, 256) == 1024, "third ladder range mismatch");
    require(server_checkpoint::ladder_min_step(1 << 30, 256) == 32768, "ladder step cap mismatch");
}

static void test_mid_prompt_gate() {
    require(!server_checkpoint::should_create_mid_prompt_checkpoint(0, 10000, false), "must not checkpoint at prompt start");
    require(server_checkpoint::should_create_mid_prompt_checkpoint(2048, 10000, false), "must allow mid-prompt checkpoint");
    require(!server_checkpoint::should_create_mid_prompt_checkpoint(9000, 10000, true), "near-end checkpoint is not a ladder checkpoint");
    require(!server_checkpoint::should_create_mid_prompt_checkpoint(10000, 10000, false), "must not checkpoint past prompt end");
}

static void test_redundant_pruning() {
    std::list<common_prompt_checkpoint> checkpoints;
    for (int64_t pos : {52, 2100, 4148, 6196, 8244}) {
        checkpoints.push_back(make_checkpoint(pos));
    }

    prune_to(checkpoints, 3);

    // The old FIFO behavior would keep [4148, 6196, 8244]. The coverage policy
    // keeps the oldest anchor and the newest checkpoint, while thinning the
    // densest interior region.
    require_eq({52, 4148, 8244}, checkpoint_positions(checkpoints));
}

static void test_size_pruning_reuses_redundant_policy() {
    std::list<common_prompt_checkpoint> checkpoints;
    for (int64_t pos : {52, 2100, 4148, 6196, 8244}) {
        checkpoints.push_back(make_checkpoint(pos, 10));
    }

    const auto pruned = server_checkpoint::prune_checkpoints_to_limit(checkpoints, 30);

    require(pruned.n_pruned == 2, "size pruning must report pruned checkpoint count");
    require(pruned.size_pruned == 20, "size pruning must report pruned checkpoint bytes");
    require_eq({52, 4148, 8244}, checkpoint_positions(checkpoints));
}

int main() {
    try {
        test_ladder_min_step();
        test_mid_prompt_gate();
        test_redundant_pruning();
        test_size_pruning_reuses_redundant_policy();
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
