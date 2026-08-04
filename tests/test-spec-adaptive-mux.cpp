//
// test-spec-adaptive-mux.cpp — unit test for the ADAPTIVE muxer (Workstream B)
//
// SCOPE
// -----
// Verifies the longest-accepted-prefix selection logic that the
// common_speculative_impl_draft_adaptive muxer uses internally after its 4
// trial verifier forwards.  The selection itself is a pure function
// (common_speculative_pick_best_accepted_idx) so the test can exercise it
// without instantiating real drafter contexts -- mocking through the
// common_speculative_impl virtual interface is heavy and architect-deferred
// (the embedded-drafter GGUF with 4 drafters is a follow-up).
//
// Test plan (no model required):
//   1. API surface: the selection helper exists, is callable, and the
//      common_speculative_type_to_str switch covers the new type.
//   2. Empty input -> index 0 (degenerate but defined behavior).
//   3. Single candidate -> index 0.
//   4. MTP has the longest prefix -> MTP wins (idx 0).
//   5. DFlash has the longest prefix -> DFlash wins (idx 1).
//   6. DSPark has the longest prefix -> DSPark wins (idx 2).
//   7. Eagle3 has the longest prefix -> Eagle3 wins (idx 3).
//   8. Tie between MTP and DFlash -> MTP wins (earlier index, the
//      architect-specified tie-break).
//   9. Tie between DFlash and Eagle3 -> DFlash wins (earlier index).
//  10. All four have the same n_accepted -> MTP wins (earliest).
//  11. Draft tokens are preserved (not consumed) by the picker.
//  12. common_speculative_pick_best_accepted_idx is a pure function: same
//      input -> same output (no hidden state, no global mutation).
//
// What this test does NOT cover (out of scope; deferred):
//   - the per-step draft() / trial_verify_one() loop in the muxer class
//   - the 4 separate llama_decode forwards + KV bookkeeping strategy
//   - the llama_memory_seq_rm rollback semantics
//   - end-to-end speculative decoding with a real model + 4 drafters
// Those need a real model and the architect-deferred embedded-drafter
// GGUF; they will be covered by integration tests when the fixtures
// land.
//

#include "common.h"
#include "speculative.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

// Convenience: build a (n_accepted, draft_tokens) pair from a token-id
// list and an accepted count.  The draft tokens are filled with simple
// monotonically increasing ids so the test can assert that the picker
// preserves the original token order.
std::pair<uint16_t, llama_tokens> draft_pair(uint16_t n_acc, int base) {
    llama_tokens tokens;
    for (int i = 0; i < 4; ++i) {
        tokens.push_back(base + i);
    }
    return { n_acc, tokens };
}

int test_api_surface() {
    // 1. common_speculative_type_to_str covers DRAFT_ADAPTIVE.
    const std::string s = common_speculative_type_to_str(COMMON_SPECULATIVE_TYPE_DRAFT_ADAPTIVE);
    if (s != "draft-adaptive") {
        std::fprintf(stderr, "expected 'draft-adaptive', got '%s'\n", s.c_str());
        return 1;
    }

    // 2. common_speculative_type_from_name round-trips.
    if (common_speculative_type_from_name("draft-adaptive") != COMMON_SPECULATIVE_TYPE_DRAFT_ADAPTIVE) {
        std::fprintf(stderr, "from_name('draft-adaptive') failed\n");
        return 2;
    }

    // 3. The picker is callable (will not compile if the symbol is missing).
    std::vector<std::pair<uint16_t, llama_tokens>> empty;
    (void) common_speculative_pick_best_accepted_idx(empty);

    return 0;
}

int test_empty_input() {
    std::vector<std::pair<uint16_t, llama_tokens>> drafts;
    const size_t idx = common_speculative_pick_best_accepted_idx(drafts);
    if (idx != 0) {
        std::fprintf(stderr, "empty: expected 0, got %zu\n", idx);
        return 10;
    }
    return 0;
}

int test_single_candidate() {
    auto drafts = { draft_pair(3, 100) };
    const size_t idx = common_speculative_pick_best_accepted_idx(drafts);
    if (idx != 0) {
        std::fprintf(stderr, "single: expected 0, got %zu\n", idx);
        return 20;
    }
    return 0;
}

int test_distinct_winners() {
    // Distinct n_accepted: 1, 4, 2, 3 -> DFlash (idx 1) wins.
    auto drafts = {
        draft_pair(1, 100),  // MTP
        draft_pair(4, 200),  // DFlash
        draft_pair(2, 300),  // DSPark
        draft_pair(3, 400),  // Eagle3
    };
    const size_t idx = common_speculative_pick_best_accepted_idx(drafts);
    if (idx != 1) {
        std::fprintf(stderr, "distinct[DFlash wins]: expected 1, got %zu\n", idx);
        return 30;
    }

    // 0, 0, 0, 5 -> Eagle3 (idx 3) wins.
    auto drafts2 = {
        draft_pair(0, 100),
        draft_pair(0, 200),
        draft_pair(0, 300),
        draft_pair(5, 400),
    };
    const size_t idx2 = common_speculative_pick_best_accepted_idx(drafts2);
    if (idx2 != 3) {
        std::fprintf(stderr, "distinct[Eagle3 wins]: expected 3, got %zu\n", idx2);
        return 31;
    }

    // 7, 0, 0, 0 -> MTP (idx 0) wins.
    auto drafts3 = {
        draft_pair(7, 100),
        draft_pair(0, 200),
        draft_pair(0, 300),
        draft_pair(0, 400),
    };
    const size_t idx3 = common_speculative_pick_best_accepted_idx(drafts3);
    if (idx3 != 0) {
        std::fprintf(stderr, "distinct[MTP wins]: expected 0, got %zu\n", idx3);
        return 32;
    }

    return 0;
}

int test_tie_break_by_order() {
    // Tie between MTP (0) and DFlash (1), same n_accepted=3:
    // MTP wins because it's earlier in the architect-specified order.
    auto drafts = {
        draft_pair(3, 100),  // MTP
        draft_pair(3, 200),  // DFlash
        draft_pair(1, 300),  // DSPark
        draft_pair(0, 400),  // Eagle3
    };
    const size_t idx = common_speculative_pick_best_accepted_idx(drafts);
    if (idx != 0) {
        std::fprintf(stderr, "tie[MTP/DFlash]: expected 0, got %zu\n", idx);
        return 40;
    }

    // Tie between DFlash (1) and Eagle3 (3), same n_accepted=2:
    // DFlash wins because it's earlier in the order.
    auto drafts2 = {
        draft_pair(1, 100),  // MTP
        draft_pair(2, 200),  // DFlash
        draft_pair(1, 300),  // DSPark
        draft_pair(2, 400),  // Eagle3
    };
    const size_t idx2 = common_speculative_pick_best_accepted_idx(drafts2);
    if (idx2 != 1) {
        std::fprintf(stderr, "tie[DFlash/Eagle3]: expected 1, got %zu\n", idx2);
        return 41;
    }

    // All four tied at n_accepted=2 -> MTP (idx 0) wins.
    auto drafts3 = {
        draft_pair(2, 100),
        draft_pair(2, 200),
        draft_pair(2, 300),
        draft_pair(2, 400),
    };
    const size_t idx3 = common_speculative_pick_best_accepted_idx(drafts3);
    if (idx3 != 0) {
        std::fprintf(stderr, "tie[all 4]: expected 0, got %zu\n", idx3);
        return 42;
    }

    // Three-way tie at 4: DFlash (1) wins over DSPark (2) and Eagle3 (3)
    // (MTP has 0, so it loses).
    auto drafts4 = {
        draft_pair(0, 100),  // MTP
        draft_pair(4, 200),  // DFlash
        draft_pair(4, 300),  // DSPark
        draft_pair(4, 400),  // Eagle3
    };
    const size_t idx4 = common_speculative_pick_best_accepted_idx(drafts4);
    if (idx4 != 1) {
        std::fprintf(stderr, "tie[3-way DFlash/DSPark/Eagle3]: expected 1, got %zu\n", idx4);
        return 43;
    }

    return 0;
}

int test_draft_tokens_preserved() {
    // The picker returns an index; the caller is responsible for looking up
    // the draft tokens.  The picker must NOT consume or modify the input.
    auto p0 = draft_pair(1, 100);
    auto p1 = draft_pair(5, 200);
    auto p2 = draft_pair(3, 300);
    auto p3 = draft_pair(4, 400);
    const auto orig_p1 = p1;  // snapshot

    std::vector<std::pair<uint16_t, llama_tokens>> drafts = { p0, p1, p2, p3 };
    const size_t idx = common_speculative_pick_best_accepted_idx(drafts);

    if (idx != 1) {
        std::fprintf(stderr, "preserved: expected 1, got %zu\n", idx);
        return 50;
    }

    // The winning draft's tokens must be intact and equal to the original.
    if (drafts[idx].second != orig_p1.second) {
        std::fprintf(stderr, "preserved: winning draft tokens were mutated\n");
        return 51;
    }
    if (drafts[idx].first != orig_p1.first) {
        std::fprintf(stderr, "preserved: winning draft n_accepted was mutated\n");
        return 52;
    }

    // And the other drafts' tokens must also be intact.
    for (size_t i = 0; i < drafts.size(); ++i) {
        if ((int) drafts[i].second.size() != 4) {
            std::fprintf(stderr, "preserved: draft[%zu] size changed\n", i);
            return 53;
        }
    }

    return 0;
}

int test_purity() {
    // Same input -> same output.  This is the "no hidden state" contract.
    auto drafts_factory = []() {
        return std::vector<std::pair<uint16_t, llama_tokens>>{
            draft_pair(2, 100),
            draft_pair(5, 200),
            draft_pair(3, 300),
            draft_pair(4, 400),
        };
    };

    auto a = drafts_factory();
    auto b = drafts_factory();
    const size_t ia = common_speculative_pick_best_accepted_idx(a);
    const size_t ib = common_speculative_pick_best_accepted_idx(b);
    if (ia != ib) {
        std::fprintf(stderr, "purity: same input produced different output (%zu vs %zu)\n", ia, ib);
        return 60;
    }
    if (ia != 1) {
        std::fprintf(stderr, "purity: expected 1, got %zu\n", ia);
        return 61;
    }

    return 0;
}

int test_order_matches_architect_spec() {
    // The architect-specified order is [MTP, DFlash, DSPark, Eagle3].
    // Verify the picker respects this by giving each one a unique
    // n_accepted that exactly one of them achieves.
    auto run = [](int mtp, int dflash, int dspark, int eagle3) {
        return common_speculative_pick_best_accepted_idx({
            draft_pair((uint16_t) mtp, 100),
            draft_pair((uint16_t) dflash, 200),
            draft_pair((uint16_t) dspark, 300),
            draft_pair((uint16_t) eagle3, 400),
        });
    };

    if (run(10, 0, 0, 0) != 0) { std::fprintf(stderr, "order: MTP failed\n");    return 70; }
    if (run(0, 10, 0, 0) != 1) { std::fprintf(stderr, "order: DFlash failed\n"); return 71; }
    if (run(0, 0, 10, 0) != 2) { std::fprintf(stderr, "order: DSPark failed\n"); return 72; }
    if (run(0, 0, 0, 10) != 3) { std::fprintf(stderr, "order: Eagle3 failed\n"); return 73; }

    return 0;
}

} // namespace

int main() {
    int rc;

    if ((rc = test_api_surface())         != 0) return rc;
    if ((rc = test_empty_input())         != 0) return rc;
    if ((rc = test_single_candidate())    != 0) return rc;
    if ((rc = test_distinct_winners())    != 0) return rc;
    if ((rc = test_tie_break_by_order())  != 0) return rc;
    if ((rc = test_draft_tokens_preserved()) != 0) return rc;
    if ((rc = test_purity())              != 0) return rc;
    if ((rc = test_order_matches_architect_spec()) != 0) return rc;

    std::printf("test-spec-adaptive-mux: all checks passed\n");
    return 0;
}
