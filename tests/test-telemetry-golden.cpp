// test-telemetry-golden.cpp
//
// Golden test for the factored spec.v1 record emitter
// common_spec_telemetry_record() in common/speculative-calibration.cpp.
//
// The emitter was moved verbatim out of the inline telemetry block in
// common_speculative_calibration_run(); this test pins the observable
// byte output so any future drift is caught:
//
//   1. A FROZEN COPY of the pre-factoring serialization (same pattern
//      as emit_spec_v1 in test-telemetry-jsonl.cpp) computes the
//      expected record from the same per-position logits rows; the
//      factored function must reproduce it byte-for-byte across a
//      matrix of shapes (topk 0 / >0 / clamped, nullptr rows, partial
//      and full acceptance).
//   2. One fully hand-computed literal record anchors the frozen copy
//      itself to absolute bytes.
//   3. provenance / sid are omitted when NULL (calibration parity) and
//      appended as trailing fields when set (runtime records).
//
// If the emission in speculative-calibration.cpp is ever changed, the
// frozen copy below must be updated in lockstep and justified.

#include "speculative-calibration.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#define TEST_ASSERT(cond)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "test-telemetry-golden: assertion failed: %s "   \
                                 "(at %s:%d)\n",                                  \
                         #cond, __FILE__, __LINE__);                              \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// Frozen copy of the pre-factoring per-step serialization, operating on
// the same per-position logits rows the factored function consumes.
// Copied from the inline block of common_speculative_calibration_run()
// as it existed before the S1 factoring; pinned here. Updated in
// lockstep with the off-by-one accept fix (confidence reads row i
// instead of row i+1; the bonus reads row n_acc instead of row n_dft).
// ---------------------------------------------------------------------------
std::string golden_spec_record(int32_t step,
                               int32_t id_last,
                               const std::vector<int32_t> & draft,
                               size_t n_acc,
                               const std::vector<const float *> & v_logits_ptrs,
                               const std::vector<const float *> & dft_logits_ptrs,
                               int32_t n_vocab,
                               int32_t topk_in,
                               const char * provenance,
                               const char * sid) {
    const int n_dft = (int) draft.size();
    const int topk  = std::min(topk_in > 0 ? topk_in : 0, n_vocab);

    auto topk_row = [&](const float * row,
                        int32_t & argmax_out) {
        std::vector<int32_t> tok;
        std::vector<float>   val;
        if (row == nullptr) {
            argmax_out = 0;
            return std::pair{std::move(tok), std::move(val)};
        }
        int32_t am = 0;
        float   am_val = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > am_val) { am_val = row[v]; am = v; }
        }
        argmax_out = am;
        if (topk == 0) {
            return std::pair{std::move(tok), std::move(val)};
        }
        std::vector<std::pair<float, int32_t>> heap;
        heap.reserve(topk);
        for (int v = 0; v < n_vocab; ++v) {
            if ((int) heap.size() < topk) {
                heap.emplace_back(row[v], v);
                std::push_heap(heap.begin(), heap.end(),
                    std::greater<std::pair<float, int32_t>>());
            } else if (row[v] > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(),
                    std::greater<std::pair<float, int32_t>>());
                heap.back() = {row[v], v};
                std::push_heap(heap.begin(), heap.end(),
                    std::greater<std::pair<float, int32_t>>());
            }
        }
        tok.reserve(topk);
        val.reserve(topk);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(),
                std::greater<std::pair<float, int32_t>>());
            tok.push_back(heap.back().second);
            val.push_back(heap.back().first);
            heap.pop_back();
        }
        return std::pair{std::move(tok), std::move(val)};
    };

    std::vector<std::vector<int32_t>> v_topk_tokens(n_dft + 1);
    std::vector<std::vector<float>>   v_topk_probs(n_dft + 1);
    std::vector<int32_t>              v_argmax_explicit(n_dft + 1, 0);
    std::vector<float>                confidence(n_dft, 0.0f);
    for (int i = 0; i <= n_dft; ++i) {
        int32_t am = 0;
        auto tk = topk_row(
            i < (int) v_logits_ptrs.size() ? v_logits_ptrs[i] : nullptr,
            am);
        v_topk_tokens[i] = std::move(tk.first);
        v_topk_probs[i]  = std::move(tk.second);
        v_argmax_explicit[i] = am;
    }
    for (int i = 0; i < n_dft; ++i) {
        const float * row = (i < (int) v_logits_ptrs.size())
                            ? v_logits_ptrs[i] : nullptr;
        if (row == nullptr) continue;
        float max_logit = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > max_logit) max_logit = row[v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            sum_exp += std::exp((double) row[v] - (double) max_logit);
        }
        const double prob = sum_exp > 0.0
            ? std::exp((double) row[draft[i]] - (double) max_logit) / sum_exp
            : 0.0;
        confidence[i] = (float) prob;
    }

    std::vector<std::vector<int32_t>> d_topk_tokens(n_dft + 1);
    std::vector<std::vector<float>>   d_topk_probs(n_dft + 1);
    std::vector<int32_t>              d_argmax_explicit(n_dft + 1, 0);
    for (int i = 0; i <= n_dft; ++i) {
        int32_t am = 0;
        auto tk = topk_row(
            i < (int) dft_logits_ptrs.size() ? dft_logits_ptrs[i] : nullptr,
            am);
        d_topk_tokens[i] = std::move(tk.first);
        d_topk_probs[i]  = std::move(tk.second);
        d_argmax_explicit[i] = am;
    }

    const int32_t bonus_local = v_argmax_explicit[n_acc];
    std::vector<int32_t> accepted_tokens;
    accepted_tokens.reserve(n_acc + 1);
    for (size_t k = 0; k < n_acc; ++k) {
        accepted_tokens.push_back(draft[k]);
    }
    if (n_acc <= (size_t) n_dft) {
        accepted_tokens.push_back(bonus_local);
    }

    std::string line;
    line  = "{\"schema\":\"llama.tessera.spec.v1\"";
    line += ",\"seq_id\":0";
    line += ",\"step_idx\":" + std::to_string(step);
    line += ",\"prime_token\":" + std::to_string(id_last);
    line += ",\"drafted\":" + std::to_string(n_dft);
    line += ",\"accepted\":" + std::to_string(n_acc);

    line += ",\"drafted_tokens\":[";
    for (int i = 0; i < n_dft; ++i) {
        if (i > 0) line += ",";
        line += std::to_string(draft[i]);
    }
    line += "]";

    line += ",\"accepted_tokens\":[";
    for (size_t i = 0; i < accepted_tokens.size(); ++i) {
        if (i > 0) line += ",";
        line += std::to_string(accepted_tokens[i]);
    }
    line += "]";

    line += ",\"confidence\":[";
    for (size_t i = 0; i < confidence.size(); ++i) {
        if (i > 0) line += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.8g",
                      (double) confidence[i]);
        line += buf;
    }
    line += "]";

    if (topk > 0) {
        line += ",\"topk\":" + std::to_string(topk);

        line += ",\"verifier_argmax\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += std::to_string(v_argmax_explicit[i]);
        }
        line += "]";

        line += ",\"drafter_argmax\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += std::to_string(d_argmax_explicit[i]);
        }
        line += "]";

        line += ",\"verifier_topk_tokens\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < v_topk_tokens[i].size(); ++k) {
                if (k > 0) line += ",";
                line += std::to_string(v_topk_tokens[i][k]);
            }
            line += "]";
        }
        line += "]";
        line += ",\"verifier_topk_probs\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < v_topk_probs[i].size(); ++k) {
                if (k > 0) line += ",";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g",
                              (double) v_topk_probs[i][k]);
                line += buf;
            }
            line += "]";
        }
        line += "]";

        line += ",\"drafter_topk_tokens\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < d_topk_tokens[i].size(); ++k) {
                if (k > 0) line += ",";
                line += std::to_string(d_topk_tokens[i][k]);
            }
            line += "]";
        }
        line += "]";
        line += ",\"drafter_topk_probs\":[";
        for (int i = 0; i <= n_dft; ++i) {
            if (i > 0) line += ",";
            line += "[";
            for (size_t k = 0; k < d_topk_probs[i].size(); ++k) {
                if (k > 0) line += ",";
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.6g",
                              (double) d_topk_probs[i][k]);
                line += buf;
            }
            line += "]";
        }
        line += "]";
    }

    if (provenance != nullptr) {
        line += ",\"provenance\":\"";
        line += provenance;
        line += "\"";
    }
    if (sid != nullptr) {
        line += ",\"sid\":\"";
        line += sid;
        line += "\"";
    }

    line += "}\n";
    return line;
}

// Deterministic LCG so the synthetic logits are reproducible.
struct lcg {
    uint32_t s = 0x2545F491u;
    float next_float() {
        s = s * 1664525u + 1013904223u;
        // map to [-4.0, 4.0)
        return ((float) (s >> 8) / (float) (1u << 24)) * 8.0f - 4.0f;
    }
};

// Builds n_rows rows of n_vocab logits; rows whose index appears in
// `null_rows` are set to nullptr to exercise the missing-row path.
struct row_set {
    std::vector<std::vector<float>> storage;
    std::vector<const float *>      ptrs;
};

row_set make_rows(lcg & rng, int n_rows, int n_vocab, const std::vector<int> & null_rows) {
    row_set rs;
    rs.storage.resize(n_rows);
    rs.ptrs.resize(n_rows, nullptr);
    for (int i = 0; i < n_rows; ++i) {
        bool is_null = false;
        for (int nr : null_rows) {
            if (nr == i) is_null = true;
        }
        if (is_null) continue;
        rs.storage[i].resize(n_vocab);
        for (int v = 0; v < n_vocab; ++v) {
            rs.storage[i][v] = rng.next_float();
        }
        rs.ptrs[i] = rs.storage[i].data();
    }
    return rs;
}

// One byte-identity case: the factored emitter must match the frozen
// pre-factoring copy exactly.
void check_case(int32_t step, int32_t id_last,
                const std::vector<int32_t> & draft, size_t n_acc,
                int n_vocab, int32_t topk,
                const std::vector<int> & v_null_rows,
                const std::vector<int> & d_null_rows,
                const char * provenance, const char * sid) {
    lcg rng_v;
    lcg rng_d;
    const int n_rows = (int) draft.size() + 1;
    row_set v = make_rows(rng_v, n_rows, n_vocab, v_null_rows);
    row_set d = make_rows(rng_d, n_rows, n_vocab, d_null_rows);

    const std::string expected = golden_spec_record(
        step, id_last, draft, n_acc, v.ptrs, d.ptrs, n_vocab, topk,
        provenance, sid);
    const std::string actual = common_spec_telemetry_record(
        step, id_last,
        llama_tokens(draft.begin(), draft.end()), n_acc,
        v.ptrs, d.ptrs, n_vocab, topk,
        provenance, sid);

    if (expected != actual) {
        std::fprintf(stderr, "test-telemetry-golden: byte mismatch\n"
                             "  expected: %s\n  actual:   %s\n",
                     expected.c_str(), actual.c_str());
        std::abort();
    }
    TEST_ASSERT(!actual.empty() && actual.back() == '\n');
}

}  // namespace

int main() {
    // -----------------------------------------------------------------
    // 1. Hand-computed literal anchor (topk == 0, n_vocab 4, 1 draft,
    //    fully accepted). Rows are exactly representable floats so the
    //    double-precision softmax math is reproducible by hand.
    //    confidence[0] is the prob of draft[0] under row 0 (the row
    //    that judges draft[0]); the bonus is argmax(row n_acc).
    //      v row 0 = [0.1, 0.2, 0.3, 0.4], draft[0] = 2
    //      confidence = e^(0.3-0.4) / (e^(0.1-0.4) + e^(0.2-0.4) +
    //                   e^(0.3-0.4) + 1) = 0.26118261... -> "%.8g"
    //      bonus = argmax(v row 1) = 3 (n_acc == n_dft here)
    // -----------------------------------------------------------------
    {
        const std::vector<float> v0 = { 0.1f, 0.2f, 0.3f, 0.4f };
        const std::vector<float> v1 = { 0.0f, 1.0f, 2.0f, 3.0f };
        const std::vector<float> d0 = { 4.0f, 3.0f, 2.0f, 1.0f };
        const std::vector<float> d1 = { 1.0f, 2.0f, 3.0f, 4.0f };
        const std::vector<const float *> v_rows = { v0.data(), v1.data() };
        const std::vector<const float *> d_rows = { d0.data(), d1.data() };

        const std::string line = common_spec_telemetry_record(
            /*step=*/0, /*id_last=*/1, /*draft=*/{ 2 }, /*n_acc=*/1,
            v_rows, d_rows, /*n_vocab=*/4, /*topk=*/0);

        const std::string expected =
            "{\"schema\":\"llama.tessera.spec.v1\",\"seq_id\":0,\"step_idx\":0,"
            "\"prime_token\":1,\"drafted\":1,\"accepted\":1,"
            "\"drafted_tokens\":[2],\"accepted_tokens\":[2,3],"
            "\"confidence\":[0.26118261]}\n";
        if (line != expected) {
            std::fprintf(stderr, "test-telemetry-golden: literal anchor mismatch\n"
                                 "  expected: %s  actual:   %s\n",
                         expected.c_str(), line.c_str());
            return 1;
        }
    }

    // -----------------------------------------------------------------
    // 1b. Off-by-one regression pin (fails on the pre-fix emitter).
    //     Row i is conditioned on prefix + draft[0..i-1] and judges
    //     draft[i]; the bonus comes from row n_acc. With n_acc < n_dft:
    //       v row 0 = [0, 4, 0, 0]  argmax 1 == draft[0] (accepted)
    //       v row 1 = [0, 0, 0, 3]  argmax 3 != draft[1] (rejected)
    //       v row 2 = [2, 0, 0, 0]  argmax 0 (row saw the rejected draft)
    //     Correct output:
    //       confidence[0] = P(1 | row 0) = e^4/(e^4+3) -> 0.94791502
    //       confidence[1] = P(2 | row 1) = 1/(3+e^3)   -> 0.043317165
    //       bonus = argmax(row n_acc=1) = 3, accepted_tokens = [1, 3]
    //     The pre-fix emitter read confidence from row i+1
    //     ([0.043317165, 0.096255139]) and the bonus from row n_dft
    //     (accepted_tokens [1, 0]).
    // -----------------------------------------------------------------
    {
        const std::vector<float> v0 = { 0.0f, 4.0f, 0.0f, 0.0f };
        const std::vector<float> v1 = { 0.0f, 0.0f, 0.0f, 3.0f };
        const std::vector<float> v2 = { 2.0f, 0.0f, 0.0f, 0.0f };
        const std::vector<float> d0 = { 1.0f, 1.0f, 1.0f, 1.0f };
        const std::vector<float> d1 = { 1.0f, 1.0f, 1.0f, 1.0f };
        const std::vector<float> d2 = { 1.0f, 1.0f, 1.0f, 1.0f };
        const std::vector<const float *> v_rows = { v0.data(), v1.data(), v2.data() };
        const std::vector<const float *> d_rows = { d0.data(), d1.data(), d2.data() };

        const std::string line = common_spec_telemetry_record(
            /*step=*/1, /*id_last=*/9, /*draft=*/{ 1, 2 }, /*n_acc=*/1,
            v_rows, d_rows, /*n_vocab=*/4, /*topk=*/0);

        const std::string expected =
            "{\"schema\":\"llama.tessera.spec.v1\",\"seq_id\":0,\"step_idx\":1,"
            "\"prime_token\":9,\"drafted\":2,\"accepted\":1,"
            "\"drafted_tokens\":[1,2],\"accepted_tokens\":[1,3],"
            "\"confidence\":[0.94791502,0.043317165]}\n";
        if (line != expected) {
            std::fprintf(stderr, "test-telemetry-golden: off-by-one regression pin mismatch\n"
                                 "  expected: %s  actual:   %s\n",
                         expected.c_str(), line.c_str());
            return 1;
        }
    }

    // -----------------------------------------------------------------
    // 2. Byte-identity matrix against the frozen pre-factoring copy.
    // -----------------------------------------------------------------
    const std::vector<int32_t> draft3 = { 17, 42, 99 };

    // topk == 0: cheap payload only.
    check_case(0, 7, draft3, /*n_acc=*/2, /*n_vocab=*/64, /*topk=*/0,
               {}, {}, nullptr, nullptr);
    // topk > 0: full per-position distributions.
    check_case(3, 29889, draft3, /*n_acc=*/1, 64, /*topk=*/8,
               {}, {}, nullptr, nullptr);
    // topk larger than the vocab: clamps to n_vocab.
    check_case(4, 5, draft3, /*n_acc=*/3, 64, /*topk=*/512,
               {}, {}, nullptr, nullptr);
    // zero acceptance.
    check_case(5, 11, draft3, /*n_acc=*/0, 64, 8, {}, {}, nullptr, nullptr);
    // empty draft (loop terminated early); n_rows == 1.
    check_case(6, 13, {}, /*n_acc=*/0, 64, 8, {}, {}, nullptr, nullptr);
    // nullptr verifier row at position 1 (confidence slot stays 0).
    check_case(7, 21, draft3, /*n_acc=*/2, 64, 8,
               /*v_null=*/{ 1 }, /*d_null=*/{}, nullptr, nullptr);
    // nullptr drafter row at the priming position.
    check_case(8, 31, draft3, /*n_acc=*/2, 64, 8,
               /*v_null=*/{}, /*d_null=*/{ 0 }, nullptr, nullptr);
    // truncated row vectors (fewer entries than n_dft+1).
    {
        lcg rng;
        row_set v = make_rows(rng, 2, 64, {});   // only rows 0..1 of 4
        row_set d = make_rows(rng, 4, 64, {});
        const std::string expected = golden_spec_record(
            9, 41, draft3, 2, v.ptrs, d.ptrs, 64, 8, nullptr, nullptr);
        const std::string actual = common_spec_telemetry_record(
            9, 41, llama_tokens(draft3.begin(), draft3.end()), 2,
            v.ptrs, d.ptrs, 64, 8, nullptr, nullptr);
        TEST_ASSERT(expected == actual);
    }

    // -----------------------------------------------------------------
    // 3. provenance / sid: omitted when NULL, appended (in that order,
    //    after the topk block) when set.
    // -----------------------------------------------------------------
    {
        lcg rng;
        row_set v = make_rows(rng, 4, 64, {});
        row_set d = make_rows(rng, 4, 64, {});
        const llama_tokens draft(draft3.begin(), draft3.end());

        const std::string bare = common_spec_telemetry_record(
            0, 7, draft, 2, v.ptrs, d.ptrs, 64, 8, nullptr, nullptr);
        TEST_ASSERT(bare.find("\"provenance\"") == std::string::npos);
        TEST_ASSERT(bare.find("\"sid\"") == std::string::npos);

        const std::string stamped = common_spec_telemetry_record(
            0, 7, draft, 2, v.ptrs, d.ptrs, 64, 8,
            "runtime", "0f3e2b1a-9c8d-4e7f-a6b5-c4d3e2f1a0b9");
        // additive only: the stamped record is the bare record with the
        // two trailing fields inserted before the closing brace.
        TEST_ASSERT(stamped.size() > bare.size());
        TEST_ASSERT(stamped.substr(0, bare.size() - 2) == bare.substr(0, bare.size() - 2));
        TEST_ASSERT(stamped.find(",\"provenance\":\"runtime\"") != std::string::npos);
        TEST_ASSERT(stamped.find(",\"sid\":\"0f3e2b1a-9c8d-4e7f-a6b5-c4d3e2f1a0b9\"") != std::string::npos);
        TEST_ASSERT(stamped.substr(stamped.size() - 2) == "}\n");
        // field order: provenance before sid, both last.
        TEST_ASSERT(stamped.find("\"provenance\"") < stamped.find("\"sid\""));

        // golden copy parity for the stamped form as well.
        const std::string expected = golden_spec_record(
            0, 7, draft3, 2, v.ptrs, d.ptrs, 64, 8,
            "runtime", "0f3e2b1a-9c8d-4e7f-a6b5-c4d3e2f1a0b9");
        TEST_ASSERT(expected == stamped);
    }

    std::fprintf(stdout, "test-telemetry-golden: OK\n");
    return 0;
}
