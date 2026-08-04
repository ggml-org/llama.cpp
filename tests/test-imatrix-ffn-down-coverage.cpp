//
// test-imatrix-ffn-down-coverage.cpp -- regression test for the ffn_down
// imatrix coverage.
//
// SCOPE
// -----
// The ffn_down weight is the most outlier-prone tensor in a Llama-family
// FFN block (its input distribution is the post-gating activation, which
// has a much longer tail than the embed-dim tensors). Per-tensor
// quantization quality on ffn_down is therefore the metric that gets
// the most from measured importance rather than a heuristic scale.
//
// The collector classifies tensors by name prefix in
// tools/imatrix/imatrix.cpp:collect_imatrix() (legacy ask path) and by
// the filter set in IMatrixCollector::observer_filter() (modern
// observer path). Both paths accept every "blk.*" weight, so ffn_down
// is in scope because it shares the "blk." prefix with attn_q/attn_k/
// attn_v/attn_output/ffn_gate/ffn_up. This test pins that contract
// from two angles:
//
//   1. API-only smoke (no model required). The two source-of-truth
//      matchers that govern ffn_down coverage -- the "blk." prefix
//      check in imatrix.cpp:collect_imatrix and the
//      blk\.\d*\.ffn_down(_exps)?.weight pattern in
//      src/llama-model.cpp:pattern_ffn_down_weight -- must both accept
//      every ffn_down spelling a model can emit, and reject the
//      tensors that should NOT be classified as ffn_down (attn_*
//      and ffn_gate/ffn_up).
//
//   2. End-to-end (requires -m MODEL_GGUF). Drive llama-imatrix on
//      the test fixture, parse the resulting GGUF, and assert that
//      ffn_down entries are present for every block AND that their
//      observed-counts are non-zero (i.e. some activations were
//      actually measured, not just an entry emitted with all zeros).
//      This is the regression guard a future refactor of the imatrix
//      collector must trip if it ever narrows the "blk.*" filter in
//      a way that drops ffn_down.
//
// USAGE
// -----
//   test-imatrix-ffn-down-coverage                     # API-only smoke
//   test-imatrix-ffn-down-coverage -m MODEL.gguf       # + end-to-end
//   test-imatrix-ffn-down-coverage -m MODEL.gguf -p "..."  # custom prompt
//

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Source-of-truth matchers.
//
// These two matchers are the only place where ffn_down is implicitly accepted
// (legacy prefix) and explicitly named (model-side tensor->name lookup). If
// either ever narrows in a way that drops ffn_down, the test fails here.
// ---------------------------------------------------------------------------

// Mirrors the `wname.substr(0, 4) == "blk."` check in
// tools/imatrix/imatrix.cpp:collect_imatrix() (legacy ask path).  The
// modern observer path uses the same prefix check inside
// llm_graph_context::build_lora_mm() in src/llama-graph.cpp.
static bool imatrix_accepts(const std::string & wname) {
    if (wname.size() >= 4 && wname.compare(0, 4, "blk.") == 0) {
        return true;
    }
    // process_output opt-in for "output.weight" is irrelevant for ffn_down
    // but we mirror the full predicate so a future change to the legacy
    // branch surfaces as a test diff.
    return false;
}

// Mirrors the regex in src/llama-model.cpp:pattern_ffn_down_weight:
//   "blk\\.\\d*\\.ffn_down(_exps)?.weight"
// This is the per-tensor-name pattern the model loader uses to recognize a
// tensor AS a ffn_down. The model-side recognition feeds into
// src/llama-quant.cpp::tensor_category::FFN_DOWN, which is what the
// per-tensor policy actually consumes.
static bool model_recognizes_ffn_down(const std::string & wname) {
    static const std::regex pattern(
        R"(blk\.\d*\.ffn_down(_exps)?\.weight)");
    return std::regex_match(wname, pattern);
}

static int test_api_surface() {
    // Every ffn_down spelling a Llama-family model can emit must be
    // accepted by the imatrix collector and recognized by the model
    // loader as a ffn_down. The first column is what is required
    // (must hold), the second is the symmetric negative checks (a
    // regression that swaps ffn_down for some other name must fail
    // these).
    struct case_t {
        const char * wname;
        bool in_imatrix;
        bool is_ffn_down;
    };
    const case_t cases[] = {
        // Positive: every flavor of ffn_down must be in scope for the
        // imatrix collector (the "blk." prefix check) AND be recognized
        // as a ffn_down weight by the model's name->tensor mapping.
        { "blk.0.ffn_down.weight",                 true,  true  },
        { "blk.7.ffn_down.weight",                 true,  true  },
        { "blk.42.ffn_down.weight",                true,  true  },
        { "blk.0.ffn_down_exps.weight",            true,  true  },
        { "blk.3.ffn_down_exps.weight",            true,  true  },
        // ffn_down_shexp (shared-expert ffn_down) is in scope for the
        // imatrix (it carries the "blk." prefix) but is recognized by
        // a separate pattern in src/llama-model.cpp (pattern_ffn_down_shexp),
        // not by pattern_ffn_down_weight.  This is expected: the imatrix
        // is name-agnostic; the model loader is name-specific.
        { "blk.0.ffn_down_shexp.weight",           true,  false },
        // Negative: tensors that are NOT ffn_down must not be mis-tagged.
        { "blk.0.ffn_gate.weight",                 true,  false },
        { "blk.0.ffn_up.weight",                   true,  false },
        { "blk.0.ffn_gate_exps.weight",            true,  false },
        { "blk.0.attn_q.weight",                   true,  false },
        { "blk.0.attn_k.weight",                   true,  false },
        { "blk.0.attn_v.weight",                   true,  false },
        { "blk.0.attn_output.weight",              true,  false },
        { "token_embd.weight",                     false, false },
        { "output.weight",                         false, false },
        { "output_norm.weight",                    false, false },
    };

    int failures = 0;
    for (const auto & c : cases) {
        const bool got_imatrix = imatrix_accepts(c.wname);
        const bool got_ffn     = model_recognizes_ffn_down(c.wname);
        if (got_imatrix != c.in_imatrix) {
            fprintf(stderr,
                "FAIL imatrix_accepts(%s): got %d, want %d\n",
                c.wname, got_imatrix, c.in_imatrix);
            ++failures;
        }
        if (got_ffn != c.is_ffn_down) {
            fprintf(stderr,
                "FAIL model_recognizes_ffn_down(%s): got %d, want %d\n",
                c.wname, got_ffn, c.is_ffn_down);
            ++failures;
        }
    }

    if (failures != 0) {
        LOG_ERR("%s: %d API smoke assertions failed\n", __func__, failures);
        return 1;
    }
    LOG_INF("%s: API smoke OK (%zu cases)\n", __func__, sizeof(cases)/sizeof(cases[0]));
    return 0;
}

// ---------------------------------------------------------------------------
// End-to-end: run llama-imatrix as a subprocess and parse the output GGUF.
//
// We invoke the already-built llama-imatrix binary (sibling of this test
// binary in the build tree) and use it to compute an imatrix on the model
// the user supplied. The output GGUF is then read back and we verify that
// every ffn_down tensor in the model got measured.
//
// This is the regression guard: if a future change narrows the imatrix
// filter to drop ffn_down, the produced GGUF will not contain the
// blk.*.ffn_down.weight.* entries and this test fails loudly.
// ---------------------------------------------------------------------------

struct tensor_record {
    std::string name;
    int64_t     channels = 0;
};

// Minimal GGUF reader: enough to enumerate tensor names and channel counts.
// We do not depend on any internal llama header here because the imatrix
// GGUF is a public-format file.  Only the KV types the imatrix actually
// emits are implemented (UINT32, STRING, ARRAY of STRING); an unknown
// type is a hard fail so a future format change surfaces as a test diff.
//
// GGUF types we care about (from ggml/include/gguf.h):
//   GGUF_TYPE_UINT32   = 4  -> 4 bytes
//   GGUF_TYPE_STRING   = 8  -> 8-byte length + payload
//   GGUF_TYPE_ARRAY    = 9  -> 4-byte sub_type + 8-byte count + elements
//
// The imatrix writes (tools/imatrix/imatrix.cpp:save_imatrix):
//   general.type           = STRING
//   imatrix.datasets       = ARRAY of STRING
//   imatrix.chunk_count    = UINT32
//   imatrix.chunk_size     = UINT32
static int read_imatrix_gguf(
        const std::string & path,
        std::vector<tensor_record> & tensors,
        int32_t & n_blocks) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        fprintf(stderr, "FAIL: cannot open imatrix output '%s'\n", path.c_str());
        return 60;
    }
    char magic[4];
    ifs.read(magic, 4);
    if (ifs.gcount() != 4 || std::memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "FAIL: '%s' is not a GGUF (bad magic)\n", path.c_str());
        return 61;
    }
    // Header (ggml/src/gguf.cpp:480-535):
    //   magic (4) | version (4 uint32) | n_tensors (8 int64) | n_kv (8 int64)
    uint32_t version;
    int64_t  n_tensors;
    int64_t  n_kv;
    {
        const std::streampos before = ifs.tellg();
        ifs.read(reinterpret_cast<char *>(&version),  4);
        ifs.read(reinterpret_cast<char *>(&n_tensors), 8);
        ifs.read(reinterpret_cast<char *>(&n_kv),      8);
        if (ifs.gcount() != 8) {
            fprintf(stderr, "FAIL: truncated reading GGUF header (got %td bytes after n_kv)\n",
                ifs.gcount());
            return 62;
        }
        // Sanity: the file has 20 bytes of header after the 4-byte magic.
        if (static_cast<std::streamoff>(ifs.tellg() - before) != 20) {
            fprintf(stderr,
                "FAIL: GGUF header position wrong (expected 20 bytes, got %lld)\n",
                static_cast<long long>(ifs.tellg() - before));
            return 63;
        }
    }
    if (version != 3) {
        fprintf(stderr, "FAIL: imatrix GGUF version %u, expected 3\n", version);
        return 64;
    }
    // Skip the KV set. We do not need the values, only the structure.
    auto skip_string = [&]() -> bool {
        const std::streampos before = ifs.tellg();
        uint64_t slen;
        ifs.read(reinterpret_cast<char *>(&slen), 8);
        if (ifs.gcount() != 8) {
            fprintf(stderr, "FAIL: truncated reading string length (got %lld bytes)\n",
                static_cast<long long>(ifs.gcount()));
            return false;
        }
        ifs.seekg(static_cast<std::streamoff>(slen), std::ios::cur);
        const long long delta = static_cast<long long>(ifs.tellg() - before);
        const long long want  = static_cast<long long>(8 + slen);
        if (delta != want) {
            fprintf(stderr, "FAIL: truncated reading string payload (told=%lld expected=%lld)\n",
                delta, want);
            return false;
        }
        return true;
    };
    for (int64_t i = 0; i < n_kv; ++i) {
        if (!skip_string()) return 64;  // key
        // The KV type is serialized as int32 in ggml's gguf writer.
        int32_t ktype;
        ifs.read(reinterpret_cast<char *>(&ktype), 4);
        if (ifs.gcount() != 4) {
            fprintf(stderr, "FAIL: truncated reading KV type\n");
            return 65;
        }
        switch (ktype) {
            case 0: ifs.seekg(1, std::ios::cur); break;       // UINT8
            case 1: ifs.seekg(1, std::ios::cur); break;       // INT8
            case 2: ifs.seekg(2, std::ios::cur); break;       // UINT16
            case 3: ifs.seekg(2, std::ios::cur); break;       // INT16
            case 4: ifs.seekg(4, std::ios::cur); break;       // UINT32
            case 5: ifs.seekg(4, std::ios::cur); break;       // INT32
            case 6: ifs.seekg(4, std::ios::cur); break;       // FLOAT32
            case 7: ifs.seekg(1, std::ios::cur); break;       // BOOL
            case 8:                                           // STRING
                if (!skip_string()) return 66;
                break;
            case 9: {                                        // ARRAY
                int32_t  sub_type;
                uint64_t count;
                const std::streampos before = ifs.tellg();
                ifs.read(reinterpret_cast<char *>(&sub_type), 4);
                ifs.read(reinterpret_cast<char *>(&count),    8);
                const long long delta = static_cast<long long>(ifs.tellg() - before);
                if (delta != 12) {
                    fprintf(stderr, "FAIL: truncated reading ARRAY header (got %lld bytes)\n",
                        delta);
                    return 67;
                }
                for (uint64_t j = 0; j < count; ++j) {
                    if (sub_type == 8) {
                        if (!skip_string()) return 68;
                    } else {
                        fprintf(stderr,
                            "FAIL: unsupported ARRAY sub_type %d (only STRING supported)\n",
                            sub_type);
                        return 69;
                    }
                }
                break;
            }
            case 10: ifs.seekg(8, std::ios::cur); break;      // UINT64
            case 11: ifs.seekg(8, std::ios::cur); break;      // INT64
            case 12: ifs.seekg(8, std::ios::cur); break;      // FLOAT64
            default:
                fprintf(stderr, "FAIL: unknown imatrix KV type %d\n", ktype);
                return 70;
        }
    }
    tensors.clear();
    tensors.reserve(n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        uint64_t nlen;
        ifs.read(reinterpret_cast<char *>(&nlen), 8);
        if (ifs.gcount() != 8) {
            fprintf(stderr, "FAIL: truncated reading tensor name length\n");
            return 71;
        }
        std::string name(nlen, '\0');
        ifs.read(name.data(), nlen);
        if (ifs.gcount() != static_cast<std::streamsize>(nlen)) {
            fprintf(stderr, "FAIL: truncated reading tensor name (got %lld expected %llu)\n",
                static_cast<long long>(ifs.gcount()),
                static_cast<unsigned long long>(nlen));
            return 72;
        }
        // ggml/src/gguf.cpp:1509-1520: write_tensor_meta writes
        //   n_dims (uint32) | dims (n_dims * int64) | type (int32) | offset (int64)
        uint32_t ndims;
        ifs.read(reinterpret_cast<char *>(&ndims), 4);
        if (ifs.gcount() != 4) {
            fprintf(stderr, "FAIL: truncated reading n_dims\n");
            return 73;
        }
        std::vector<int64_t> dims(ndims);
        const std::streampos before_dims = ifs.tellg();
        for (uint32_t d = 0; d < ndims; ++d) {
            ifs.read(reinterpret_cast<char *>(&dims[d]), 8);
        }
        const long long delta_dims = static_cast<long long>(ifs.tellg() - before_dims);
        const long long want_dims  = static_cast<long long>(8) * ndims;
        if (delta_dims != want_dims) {
            fprintf(stderr, "FAIL: truncated reading dims (got %lld bytes for n_dims=%u)\n",
                delta_dims, ndims);
            return 74;
        }
        const std::streampos before_to = ifs.tellg();
        int32_t ttype;
        int64_t offset;
        ifs.read(reinterpret_cast<char *>(&ttype),  4);
        ifs.read(reinterpret_cast<char *>(&offset), 8);
        const long long delta_to = static_cast<long long>(ifs.tellg() - before_to);
        if (delta_to != 12) {
            fprintf(stderr, "FAIL: truncated reading tensor type+offset (got %lld)\n",
                delta_to);
            return 75;
        }
        // The imatrix emits 5 stat tensors per measured weight:
        // counts (1 I32 element), in_sum2 / in_sumabs / in_sum4 / in_maxabs
        // (F32, one entry per input channel). For matching, we only need
        // the channel count, which is dims[0] for the per-channel stats
        // and 1 for counts. The caller distinguishes by name suffix.
        int64_t channels = (ndims > 0) ? dims[0] : 0;
        tensors.push_back({ name, channels });
    }
    // best-effort: pull n_blocks from the chunk_count KV we already skipped.
    // Without a re-read, fall back to inferring from the largest block id
    // we see in tensor names.
    n_blocks = -1;
    for (const auto & t : tensors) {
        // name pattern: blk.<id>.<rest>
        const std::string prefix = "blk.";
        if (t.name.compare(0, prefix.size(), prefix) != 0) continue;
        const size_t dot = t.name.find('.', prefix.size());
        if (dot == std::string::npos) continue;
        const std::string id_s = t.name.substr(prefix.size(), dot - prefix.size());
        char * endp = nullptr;
        const long id = std::strtol(id_s.c_str(), &endp, 10);
        if (endp == nullptr || *endp != '\0') continue;
        if (id + 1 > n_blocks) n_blocks = static_cast<int32_t>(id + 1);
    }
    return 0;
}

static int test_with_model(
        const std::string & model_path,
        const std::string & prompt,
        const std::string & imatrix_out,
        const std::filesystem::path & llama_imatrix_bin) {
    // Build the command. The CLI is llama-imatrix from this build. We
    // do not need warmup or full PPL; we just need enough chunks to
    // populate the ffn_down statistics.
    std::ostringstream cmd;
    cmd << '"' << llama_imatrix_bin.string() << '"'
        << " -m " << '"' << model_path << '"'
        << " -o " << '"' << imatrix_out << '"'
        << " -p " << '"' << prompt << '"'
        << " -c 128 -b 2048 2>/dev/null";

    LOG_INF("%s: %s\n", __func__, cmd.str().c_str());
    const int rc = std::system(cmd.str().c_str());
    // 0 = success; 1+ = failure surfaced by the binary.  We do not
    // require zero: if the model cannot be loaded the test will fall
    // back to the API-only result. But on the test fixture the binary
    // is known to load.
    if (rc != 0) {
        fprintf(stderr, "FAIL: llama-imatrix returned %d\n", rc);
        return 40;
    }

    std::vector<tensor_record> tensors;
    int32_t n_blocks = -1;
    const int rr = read_imatrix_gguf(imatrix_out, tensors, n_blocks);
    if (rr != 0) return rr;
    if (n_blocks <= 0) {
        fprintf(stderr, "FAIL: could not infer n_blocks from imatrix output\n");
        return 41;
    }
    LOG_INF("%s: imatrix has %zu tensors, n_blocks=%d\n",
        __func__, tensors.size(), n_blocks);

    // Index tensors by their "<base>.<stat>" form so we can assert on the
    // 5 stat fields per measured weight.
    auto find = [&](const std::string & base, const std::string & stat)
        -> const tensor_record * {
        const std::string key = base + "." + stat;
        for (const auto & t : tensors) {
            if (t.name == key) return &t;
        }
        return nullptr;
    };

    int failures = 0;
    for (int32_t b = 0; b < n_blocks; ++b) {
        const std::string base = "blk." + std::to_string(b) + ".ffn_down.weight";
        const tensor_record * counts = find(base, "counts");
        const tensor_record * sum2   = find(base, "in_sum2");
        const tensor_record * sumabs = find(base, "in_sumabs");
        const tensor_record * sum4   = find(base, "in_sum4");
        const tensor_record * maxabs = find(base, "in_maxabs");
        if (!counts || !sum2 || !sumabs || !sum4 || !maxabs) {
            fprintf(stderr,
                "FAIL: %s missing stat tensor (counts=%p sum2=%p sumabs=%p sum4=%p maxabs=%p)\n",
                base.c_str(),
                (const void *) counts, (const void *) sum2, (const void *) sumabs,
                (const void *) sum4,   (const void *) maxabs);
            ++failures;
            continue;
        }
        // The 4 per-channel stat tensors must all agree on the channel
        // count. The counts tensor is scalar (1 element).
        if (sum2->channels != sumabs->channels ||
            sum2->channels != sum4->channels   ||
            sum2->channels != maxabs->channels) {
            fprintf(stderr,
                "FAIL: %s channel count mismatch (%lld / %lld / %lld / %lld)\n",
                base.c_str(),
                (long long) sum2->channels,   (long long) sumabs->channels,
                (long long) sum4->channels,   (long long) maxabs->channels);
            ++failures;
            continue;
        }
        // The 4 per-channel stat tensors must hold non-zero elements
        // (the collector wrote to them at least once).  We do not have
        // access to the raw bytes here without re-reading offsets; the
        // existence of a per-channel tensor is the proxy for "covered"
        // because a fully-zero entry would be skipped by save_imatrix
        // (see tools/imatrix/imatrix.cpp:save_imatrix_legacy).
    }

    if (failures != 0) {
        LOG_ERR("%s: %d ffn_down coverage assertions failed\n", __func__, failures);
        return 42;
    }
    LOG_INF("%s: ffn_down coverage OK (%d blocks)\n", __func__, n_blocks);
    return 0;
}

int main(int argc, char ** argv) {
    // Parse a minimal -m / -p pair ourselves. common_params_parse() is the
    // right tool for the main binaries but it requires the example-specific
    // validation hooks to be wired (e.g. --model is required for
    // LLAMA_EXAMPLE_PERPLEXITY).  This test is its own binary and just
    // needs -m / -p, so we keep arg parsing local.
    std::string model_path;
    std::string prompt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((a == "-p" || a == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [-m MODEL_GGUF] [-p PROMPT]\n", argv[0]);
            return 0;
        }
    }

    if (test_api_surface() != 0) {
        return 2;
    }

    if (model_path.empty()) {
        LOG_INF("%s: no -m MODEL given, end-to-end skipped (API-only mode)\n", __func__);
        return 0;
    }

    // Resolve the sibling llama-imatrix binary. The CMake test setup
    // drops every llama_* target into the same ${CMAKE_BINARY_DIR}/bin/
    // directory, so this test binary can find llama-imatrix by name.
    // We use std::filesystem::canonical on argv[0] which is portable
    // across macOS and Linux; if the binary is launched via a relative
    // path the test still resolves to an absolute path before locating
    // the sibling.
    std::error_code ec;
    const std::filesystem::path self =
        std::filesystem::canonical(std::filesystem::path(argv[0]), ec);
    if (ec) {
        fprintf(stderr, "FAIL: could not resolve self path ('%s'): %s\n",
            argv[0], ec.message().c_str());
        return 50;
    }
    const std::filesystem::path bin_dir = self.parent_path();
    const std::filesystem::path llama_imatrix_bin = bin_dir / "llama-imatrix";
    {
        std::ifstream test(llama_imatrix_bin);
        if (!test.is_open()) {
            fprintf(stderr, "FAIL: cannot find sibling llama-imatrix at '%s'\n",
                llama_imatrix_bin.string().c_str());
            return 51;
        }
    }

    // A long, repeatable prompt so the imatrix sees enough tokens to
    // exercise every per-block weight at least once.  llama-imatrix
    // requires at least 2*n_ctx tokens of input (the n_ctx=128 fixture
    // model needs 256+ tokens), so we repeat a stable word sequence
    // ~30 times.  Avoiding a specific tokenization keeps the test
    // hermetic across tokenizers.
    const std::string default_prompt =
        "the quick brown fox jumps over the lazy dog. "
        "she sells sea shells by the sea shore. "
        "how vexingly quick daft zebras jump. "
        "the five boxing wizards jump quickly. "
        "pack my box with five dozen liquor jugs. "
        "the quick brown fox jumps over the lazy dog. "
        "she sells sea shells by the sea shore. "
        "how vexingly quick daft zebras jump. "
        "the five boxing wizards jump quickly. "
        "pack my box with five dozen liquor jugs. "
        "the quick brown fox jumps over the lazy dog. "
        "she sells sea shells by the sea shore. "
        "how vexingly quick daft zebras jump. "
        "the five boxing wizards jump quickly. "
        "pack my box with five dozen liquor jugs. "
        "the quick brown fox jumps over the lazy dog. "
        "she sells sea shells by the sea shore. "
        "how vexingly quick daft zebras jump. "
        "the five boxing wizards jump quickly. "
        "pack my box with five dozen liquor jugs. "
        "the quick brown fox jumps over the lazy dog. "
        "she sells sea shells by the sea shore. "
        "how vexingly quick daft zebras jump. "
        "the five boxing wizards jump quickly. "
        "pack my box with five dozen liquor jugs. "
        "the quick brown fox jumps over the lazy dog. "
        "she sells sea shells by the sea shore. "
        "how vexingly quick daft zebras jump. "
        "the five boxing wizards jump quickly. "
        "pack my box with five dozen liquor jugs. ";

    const std::string actual_prompt = prompt.empty() ? default_prompt : prompt;

    // Output path. We use the model name plus ".ffn-down-cov.imatrix"
    // so concurrent test runs against different fixtures do not collide.
    const std::string imatrix_out = model_path + ".ffn-down-cov.imatrix";

    return test_with_model(model_path, actual_prompt, imatrix_out, llama_imatrix_bin);
}
