#include "testing.h"

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

// find_slot reports position inconsistencies only through LLAMA_LOG_WARN,
// so the tests observe them through the public log callback
static std::string g_warnings;

static void capture_warnings(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_WARN) {
        g_warnings += text;
    }
}

// a recurrent cache over an empty model: find_slot drives only the per-sequence
// cells, so no layers or state tensors are needed
struct test_memory {
    llama_model * model = nullptr;
    llama_memory_recurrent mem;

    test_memory(uint32_t n_seq_max = 4)
        : model(llama_model_create(LLM_ARCH_MAMBA, llama_model_default_params())),
          mem(*model, GGML_TYPE_F32, GGML_TYPE_F32, false, std::max((uint32_t) 1, n_seq_max), n_seq_max, 0, nullptr) {}

    ~test_memory() {
        delete model;
    }
};

// fill the per-token sequence ids the same way llama_batch_allocr::ubatch_add does
static void fill_seq_ids(llama_ubatch & ub, llama_seq_id seq_id) {
    ub.data->seq_id_data.assign(ub.n_tokens, seq_id);
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        ub.n_seq_id[i] = 1;
        ub.seq_id[i]   = &ub.data->seq_id_data[i];
    }
}

// token-input ubatch with the 1d positions broadcast to every M-RoPE section,
// like llama_batch_allocr::ubatch_add does for text batches
static llama_ubatch make_ubatch_text(llama_batch_allocr & ba, const std::vector<llama_pos> & pos) {
    llama_ubatch ub = ba.ubatch_reserve((uint32_t) pos.size(), 1);

    for (size_t i = 0; i < pos.size(); ++i) {
        ub.token[i]  = 1;
        ub.output[i] = i + 1 == pos.size();
        for (uint32_t j = 0; j < ub.n_pos; ++j) {
            ub.pos[j*ub.n_tokens + i] = pos[i];
        }
    }

    fill_seq_ids(ub, 0);

    return ub;
}

// embd-input ubatch for the image tokens [i0, i1) of an nx*ny grid, laid out like
// mtmd_helper: every image token shares the temporal position (section 0) while
// sections 1/2 walk the grid rows/columns; splitting the image across ubatches
// keeps the same positions (decode_embd_batch::get_view)
static llama_ubatch make_ubatch_image(llama_batch_allocr & ba, llama_pos pos_0, uint32_t nx, uint32_t ny, uint32_t i0, uint32_t i1) {
    const uint32_t n_tokens = i1 - i0;

    llama_ubatch ub = ba.ubatch_reserve(n_tokens, 1);

    GGML_ASSERT(ub.n_pos == 4 && i0 < i1 && i1 <= nx*ny);

    // image tokens are decoded as embeddings
    ub.data->embd.resize(n_tokens);
    ub.token = nullptr;
    ub.embd  = ub.data->embd.data();

    for (uint32_t k = 0; k < n_tokens; ++k) {
        const uint32_t i = i0 + k;
        ub.pos[                 k] = pos_0;                        // temporal: shared by the whole image
        ub.pos[1*ub.n_tokens + k] = pos_0 + (llama_pos) (i / nx); // y
        ub.pos[2*ub.n_tokens + k] = pos_0 + (llama_pos) (i % nx); // x
        ub.pos[3*ub.n_tokens + k] = 0;                            // unused
        ub.output[k] = k + 1 == n_tokens;
    }

    fill_seq_ids(ub, 0);

    return ub;
}

// decode one ubatch through find_slot and require it to stay silent
static void check_slot(testing & t, test_memory & tm, const llama_ubatch & ub, const char * what, llama_pos cell_pos) {
    g_warnings.clear();
    t.assert_true(std::string(what) + ": find_slot failed", tm.mem.find_slot(ub));
    t.assert_true(std::string(what) + ": unexpected warning" + (g_warnings.empty() ? "" : ": " + g_warnings), g_warnings.empty());
    t.assert_equal(std::string(what) + ": cell pos", cell_pos, tm.mem.cells[0].pos);
}

// regression test for the M-RoPE position checks in find_slot: image ubatches
// share one temporal position and the text after an image resumes at
// pos + max(nx, ny), so only a backtrack is inconsistent (see #28166)
static void test_mrope(testing & t) {
    // a 3x2 image decoded between two text chunks
    const uint32_t nx = 3;
    const uint32_t ny = 2;
    const uint32_t n_img = nx*ny;
    const llama_pos pos_img = 6;
    const llama_pos pos_txt = pos_img + (llama_pos) std::max(nx, ny); // text resumes at pos + max(nx, ny)

    test_memory tm;
    llama_batch_allocr ba(4); // M-RoPE positions

    // text before the image: positions 0..5
    {
        llama_ubatch ub = make_ubatch_text(ba, {0, 1, 2, 3, 4, 5});
        check_slot(t, tm, ub, "text prefill", 5);
    }

    // the image embeddings, split into two ubatches like llama_decode does when
    // the image does not fit into a single ubatch: both share the temporal
    // position 6, the second one overlapping the last stored position
    for (uint32_t u = 0; u < 2; ++u) {
        llama_ubatch ub = make_ubatch_image(ba, pos_img, nx, ny, u*(n_img/2), (u + 1)*(n_img/2));
        t.assert_true("image ubatch: pos is 2d", ub.is_pos_2d());
        check_slot(t, tm, ub, "image ubatch", pos_img);
    }

    // the first text ubatch after the image jumps to pos + max(nx, ny)
    {
        std::vector<llama_pos> pos;
        for (llama_pos p = pos_txt; p < pos_txt + 4; ++p) {
            pos.push_back(p);
        }
        llama_ubatch ub = make_ubatch_text(ba, pos);
        check_slot(t, tm, ub, "text after image", pos_txt + 3);
    }

    // negative control: with 2d positions a token input must start strictly past
    // the last stored position - re-decoding position 12 must warn
    {
        llama_ubatch ub = make_ubatch_text(ba, {12});
        g_warnings.clear();
        t.assert_true(tm.mem.find_slot(ub));
        t.assert_true("token backtrack: warning expected", g_warnings.find("backtracking token position") != std::string::npos);
    }
}

// the 1d path keeps the strictly-consecutive check
static void test_pos_1d(testing & t) {
    test_memory tm;
    llama_batch_allocr ba(1);

    {
        llama_ubatch ub = make_ubatch_text(ba, {0, 1, 2, 3, 4, 5});
        check_slot(t, tm, ub, "consecutive text", 5);
    }

    {
        llama_ubatch ub = make_ubatch_text(ba, {6, 7});
        check_slot(t, tm, ub, "consecutive continuation", 7);
    }

    // negative control: skipping a position must still warn
    {
        llama_ubatch ub = make_ubatch_text(ba, {10});
        g_warnings.clear();
        t.assert_true(tm.mem.find_slot(ub));
        t.assert_true("1d gap: warning expected", g_warnings.find("non-consecutive token position") != std::string::npos);
    }
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    // route the library warnings to the capture instead of stderr
    llama_log_set(capture_warnings, nullptr);

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    t.test("mrope",   test_mrope);
    t.test("pos_1d",  test_pos_1d);

    return t.summary();
}
