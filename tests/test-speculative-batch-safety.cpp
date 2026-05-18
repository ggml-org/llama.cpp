// Unit tests for the batch-safety helpers used by MTP speculative decoding.
//
// These exist primarily to lock in the contract that batches which are unsafe
// for MTP processing (image embedding batches, batches without token ids,
// batches with non-consecutive or interleaved sequence rows, ...) are rejected
// up-front so the MTP draft context cannot crash on them.
//
// The helpers live in common/speculative.cpp and are exposed via
// common/speculative.h for test access. They are otherwise used internally to
// gate `common_speculative_state_draft_mtp::process()`.

#include "llama.h"
#include "speculative.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Holds owning storage for a llama_batch test fixture so the pointer arrays
// stay alive for the duration of a single check call.
struct batch_fixture {
    std::vector<llama_token>                     tokens;
    std::vector<float>                           embd;
    std::vector<llama_pos>                       pos;
    std::vector<int32_t>                         n_seq_id;
    std::vector<std::vector<llama_seq_id>>       seq_id_storage;
    std::vector<llama_seq_id *>                  seq_id_ptrs;

    llama_batch batch{};

    // Build a basic token-only batch:
    //   - one sequence per row (seq_id supplied or all 0 if empty)
    //   - positions supplied or generated 0..n-1
    void build_token_batch(const std::vector<llama_token> & toks,
                           const std::vector<llama_pos>   & positions = {},
                           const std::vector<llama_seq_id> & seq_ids  = {}) {
        tokens = toks;
        const int32_t n = (int32_t) tokens.size();

        if (positions.empty()) {
            pos.resize(n);
            for (int32_t i = 0; i < n; ++i) {
                pos[i] = i;
            }
        } else {
            pos = positions;
        }

        n_seq_id.assign(n, 1);
        seq_id_storage.clear();
        seq_id_storage.resize(n);
        seq_id_ptrs.assign(n, nullptr);
        for (int32_t i = 0; i < n; ++i) {
            const llama_seq_id sid = seq_ids.empty() ? 0 : seq_ids[i];
            seq_id_storage[i] = { sid };
            seq_id_ptrs[i]    = seq_id_storage[i].data();
        }

        batch          = {};
        batch.n_tokens = n;
        batch.token    = tokens.data();
        batch.embd     = nullptr;
        batch.pos      = pos.data();
        batch.n_seq_id = n_seq_id.data();
        batch.seq_id   = seq_id_ptrs.data();
    }
};

void expect_safe(bool ok, const std::string & reason, const char * name) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s expected SAFE, got UNSAFE (%s)\n", name, reason.c_str());
        std::abort();
    }
    std::fprintf(stderr, "  ok: %s -> SAFE\n", name);
}

void expect_unsafe(bool ok, const std::string & reason, const char * name) {
    if (ok) {
        std::fprintf(stderr, "FAIL: %s expected UNSAFE, got SAFE\n", name);
        std::abort();
    }
    if (reason.empty()) {
        std::fprintf(stderr, "FAIL: %s reason was empty\n", name);
        std::abort();
    }
    std::fprintf(stderr, "  ok: %s -> UNSAFE (%s)\n", name, reason.c_str());
}

void test_empty_batch() {
    batch_fixture f;
    f.batch          = {};
    f.batch.n_tokens = 0;

    std::string reason;
    expect_unsafe(common_speculative_batch_is_token_only(f.batch, &reason), reason,
                  "empty batch (token_only)");
    reason.clear();
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "empty batch (mtp_safe)");
}

void test_null_token_pointer() {
    batch_fixture f;
    f.build_token_batch({ 1, 2, 3 });
    f.batch.token = nullptr;  // simulate an embeddings-only batch shape

    std::string reason;
    expect_unsafe(common_speculative_batch_is_token_only(f.batch, &reason), reason,
                  "null token ptr (token_only)");
    reason.clear();
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "null token ptr (mtp_safe)");
}

void test_embedding_batch() {
    batch_fixture f;
    f.build_token_batch({ 1, 2, 3 });
    // simulate a multimodal embedding batch by attaching an embd buffer
    f.embd.assign(3, 0.0f);
    f.batch.embd = f.embd.data();

    std::string reason;
    expect_unsafe(common_speculative_batch_is_token_only(f.batch, &reason), reason,
                  "embedding batch (token_only)");
    reason.clear();
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "embedding batch (mtp_safe)");
}

void test_single_seq_consecutive() {
    batch_fixture f;
    f.build_token_batch({ 10, 11, 12, 13 }, /*positions=*/ { 5, 6, 7, 8 });

    std::string reason;
    expect_safe(common_speculative_batch_is_token_only(f.batch, &reason), reason,
                "single seq consecutive (token_only)");
    reason.clear();
    expect_safe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                "single seq consecutive (mtp_safe)");
}

void test_single_seq_non_consecutive() {
    batch_fixture f;
    f.build_token_batch({ 10, 11, 12 }, /*positions=*/ { 0, 1, 3 }); // gap at 2

    std::string reason;
    // token-only check is permissive: it accepts the batch shape
    expect_safe(common_speculative_batch_is_token_only(f.batch, &reason),
                reason, "non-consecutive (token_only is permissive)");
    reason.clear();
    // mtp-safe check is strict: it must reject
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "single seq non-consecutive (mtp_safe)");
}

void test_multi_seq_per_seq_consecutive_not_interleaved() {
    // Two sequences, each consecutive within itself, presented in contiguous
    // blocks. Conservative policy should accept this since rows are not
    // interleaved.
    batch_fixture f;
    f.build_token_batch(
        /*tokens=*/    { 10, 11, 20, 21 },
        /*positions=*/ {  0,  1,  0,  1 },
        /*seq_ids=*/   {  0,  0,  1,  1 });

    std::string reason;
    expect_safe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                "multi seq, contiguous blocks, per-seq consecutive (mtp_safe)");
}

void test_multi_seq_interleaved_rejected() {
    // Rows interleave [0,0,1,0]. Spec says reject this for current MTP.
    batch_fixture f;
    f.build_token_batch(
        /*tokens=*/    { 10, 11, 20, 12 },
        /*positions=*/ {  0,  1,  0,  2 },
        /*seq_ids=*/   {  0,  0,  1,  0 });

    std::string reason;
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "multi seq interleaved (mtp_safe)");
}

void test_n_seq_id_not_one_rejected() {
    // Row claiming multiple seq ids must be rejected by the MTP-safe check.
    batch_fixture f;
    f.build_token_batch({ 10, 11 });
    // override second row to claim 2 seq ids
    f.seq_id_storage[1] = { 0, 1 };
    f.seq_id_ptrs[1]    = f.seq_id_storage[1].data();
    f.n_seq_id[1]       = 2;

    std::string reason;
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "row with n_seq_id != 1 (mtp_safe)");
}

void test_negative_seq_id_rejected() {
    batch_fixture f;
    f.build_token_batch({ 10, 11 }, {}, { 0, -1 });

    std::string reason;
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "negative seq id (mtp_safe)");
}

void test_negative_position_rejected() {
    batch_fixture f;
    f.build_token_batch({ 10, 11 }, /*positions=*/ { 0, -1 });

    std::string reason;
    expect_unsafe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                  "negative position (mtp_safe)");
}

void test_missing_seq_id_arrays_rejected() {
    batch_fixture f;
    f.build_token_batch({ 10, 11 });
    f.batch.seq_id   = nullptr;

    std::string reason;
    expect_unsafe(common_speculative_batch_is_token_only(f.batch, &reason), reason,
                  "missing seq_id arr (token_only)");
}

void test_single_token_batch_safe() {
    batch_fixture f;
    f.build_token_batch({ 42 });

    std::string reason;
    expect_safe(common_speculative_batch_is_mtp_process_safe(f.batch, &reason), reason,
                "single-token batch (mtp_safe)");
}

} // namespace

int main(int /*argc*/, char ** /*argv*/) {
    std::fprintf(stderr, "test-speculative-batch-safety: starting\n");

    test_empty_batch();
    test_null_token_pointer();
    test_embedding_batch();
    test_single_seq_consecutive();
    test_single_seq_non_consecutive();
    test_multi_seq_per_seq_consecutive_not_interleaved();
    test_multi_seq_interleaved_rejected();
    test_n_seq_id_not_one_rejected();
    test_negative_seq_id_rejected();
    test_negative_position_rejected();
    test_missing_seq_id_arrays_rejected();
    test_single_token_batch_safe();

    std::fprintf(stderr, "test-speculative-batch-safety: all assertions passed\n");
    return 0;
}
