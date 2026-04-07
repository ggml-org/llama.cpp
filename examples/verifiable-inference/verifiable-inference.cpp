#include "llama.h"

#include "../../common/common.h"
#include "../../common/download.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#include "sha256/sha256.h"
}

namespace {

static std::string hex_encode(const uint8_t * data, size_t n) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string           out;
    out.resize(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out[2 * i + 0] = hex[(data[i] >> 4) & 0xF];
        out[2 * i + 1] = hex[(data[i] >> 0) & 0xF];
    }
    return out;
}

static std::string safe_file_component(std::string s) {
    for (char & c : s) {
        const bool ok = (c >= '0' && c <= '9') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= 'a' && c <= 'z') ||
                        c == '.' || c == '-' || c == '_';
        if (!ok) {
            c = '_';
        }
    }
    if (s.empty()) {
        s = "tensor";
    }
    return s;
}

static std::array<uint8_t, SHA256_DIGEST_SIZE> sha256_bytes(const void * data, size_t n) {
    std::array<uint8_t, SHA256_DIGEST_SIZE> out{};
    sha256_hash(out.data(), (const unsigned char *) data, n);
    return out;
}

static std::array<uint8_t, SHA256_DIGEST_SIZE> sha256_concat(const std::array<uint8_t, SHA256_DIGEST_SIZE> & a,
                                                             const std::array<uint8_t, SHA256_DIGEST_SIZE> & b) {
    uint8_t buf[SHA256_DIGEST_SIZE * 2];
    memcpy(buf, a.data(), SHA256_DIGEST_SIZE);
    memcpy(buf + SHA256_DIGEST_SIZE, b.data(), SHA256_DIGEST_SIZE);
    return sha256_bytes(buf, sizeof(buf));
}

struct vi_params {
    std::string model_path;
    std::string hf_repo;
    std::string hf_file;
    std::string hf_token;
    bool        offline = false;
    std::string prompt;
    int32_t     n_gpu_layers = 0;
    uint32_t    seed         = 0;

    std::string              out_dir   = "vi-out";
    int32_t                  n_samples = 16;
    std::vector<std::string> tensor_filters;  // regex
};

static void print_usage(const char * prog) {
    std::fprintf(stderr,
                 "usage: %s (-m MODEL.gguf | --hf-repo REPO [--hf-file FILE]) -p PROMPT [options]\n"
                 "\n"
                 "options:\n"
                 "  -m, --model PATH              model path (GGUF)\n"
                 "      --hf-repo REPO[:TAG]       download from Hugging Face repo (e.g. ggml-org/models)\n"
                 "      --hf-file PATH             specific file in the HF repo (e.g. tinyllamas/stories15M-q4_0.gguf)\n"
                 "      --hf-token TOKEN           HF token (optional, for gated repos)\n"
                 "      --offline                  do not use network; only use cached HF files\n"
                 "  -p, --prompt TEXT             prompt\n"
                 "  -ngl, --n-gpu-layers N        number of layers to offload (default: 0)\n"
                 "      --vi-out DIR              output directory (default: vi-out)\n"
                 "      --vi-samples N            number of openings to produce (default: 16)\n"
                 "      --vi-seed N               seed for challenge derivation (default: 0)\n"
                 "      --vi-tensor-filter REGEX  restrict trace tensors by name (can repeat)\n",
                 prog);
}

static bool parse_args(int argc, char ** argv, vi_params & p) {
    auto need = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "-m" || arg == "--model") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.model_path = v;
            continue;
        }
        if (arg == "--hf-repo") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.hf_repo = v;
            continue;
        }
        if (arg == "--hf-file") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.hf_file = v;
            continue;
        }
        if (arg == "--hf-token") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.hf_token = v;
            continue;
        }
        if (arg == "--offline") {
            p.offline = true;
            continue;
        }
        if (arg == "-p" || arg == "--prompt") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.prompt = v;
            continue;
        }
        if (arg == "-ngl" || arg == "--n-gpu-layers") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.n_gpu_layers = std::atoi(v);
            continue;
        }
        if (arg == "--vi-out") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.out_dir = v;
            continue;
        }
        if (arg == "--vi-samples") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.n_samples = std::max(0, std::atoi(v));
            continue;
        }
        if (arg == "--vi-seed") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.seed = (uint32_t) std::strtoul(v, nullptr, 10);
            continue;
        }
        if (arg == "--vi-tensor-filter") {
            const char * v = need(i);
            if (!v) {
                return false;
            }
            p.tensor_filters.emplace_back(v);
            continue;
        }

        std::fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
        return false;
    }

    if ((p.model_path.empty() && p.hf_repo.empty()) || p.prompt.empty()) {
        return false;
    }

    return true;
}

struct trace_entry_meta {
    std::string                        name;
    std::string                        op_desc;
    std::string                        type_name;
    std::array<int64_t, GGML_MAX_DIMS> ne{};
    size_t                             nbytes = 0;
};

struct merkle_tree {
    // levels[0] = leaves, levels.back()[0] = root
    std::vector<std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>>> levels;

    std::array<uint8_t, SHA256_DIGEST_SIZE> root() const {
        assert(!levels.empty());
        assert(!levels.back().empty());
        return levels.back().front();
    }
};

static std::array<uint8_t, SHA256_DIGEST_SIZE> merkle_recompute_root(
    size_t leaf_index,
    std::array<uint8_t, SHA256_DIGEST_SIZE> cur,
    const std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> & siblings) {
    size_t idx = leaf_index;
    for (const auto & sib : siblings) {
        // Our tree combines as H(left || right). Direction is determined by idx parity at this level.
        cur = (idx % 2 == 0) ? sha256_concat(cur, sib) : sha256_concat(sib, cur);
        idx /= 2;
    }
    return cur;
}

static merkle_tree build_merkle(std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> leaves) {
    merkle_tree mt;
    mt.levels.emplace_back(std::move(leaves));
    while (mt.levels.back().size() > 1) {
        const auto &                                         cur = mt.levels.back();
        std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> nxt;
        nxt.reserve((cur.size() + 1) / 2);
        for (size_t i = 0; i < cur.size(); i += 2) {
            const auto & L = cur[i];
            const auto & R = (i + 1 < cur.size()) ? cur[i + 1] : cur[i];
            nxt.push_back(sha256_concat(L, R));
        }
        mt.levels.emplace_back(std::move(nxt));
    }
    return mt;
}

static std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> merkle_proof(const merkle_tree & mt, size_t leaf_idx) {
    std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> path;
    size_t                                               idx = leaf_idx;
    for (size_t lvl = 0; lvl + 1 < mt.levels.size(); ++lvl) {
        const auto & cur = mt.levels[lvl];
        const size_t sib = (idx ^ 1);
        if (sib < cur.size()) {
            path.push_back(cur[sib]);
        } else {
            path.push_back(cur[idx]);  // duplicated last leaf in odd case
        }
        idx /= 2;
    }
    return path;
}

static bool tensor_name_matches(const char * name, const std::vector<std::regex> & filters) {
    if (filters.empty()) {
        return true;
    }
    for (const auto & r : filters) {
        if (std::regex_search(name, r)) {
            return true;
        }
    }
    return false;
}

struct vi_trace_commit_data {
    std::vector<std::regex> filters;

    // temporary buffer for device->host copies
    std::vector<uint8_t> scratch;

    // committed trace
    std::vector<trace_entry_meta>                        meta;
    std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> leaf_hashes;
};

static std::array<uint8_t, SHA256_DIGEST_SIZE> hash_trace_leaf(const trace_entry_meta & m,
                                                               const uint8_t *          bytes,
                                                               size_t                   nbytes) {
    // A simple domain-separated encoding:
    // H( "llama.cpp/vi-leaf/v1" || name || 0 || op || 0 || type || 0 || ne[0..3] || nbytes || raw_bytes )
    std::ostringstream oss;
    oss << "llama.cpp/vi-leaf/v1" << '\0' << m.name << '\0' << m.op_desc << '\0' << m.type_name << '\0' << m.ne[0]
        << "," << m.ne[1] << "," << m.ne[2] << "," << m.ne[3] << '\0' << nbytes << '\0';
    const std::string hdr = oss.str();

    sha256_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const unsigned char *) hdr.data(), hdr.size());
    sha256_update(&ctx, (const unsigned char *) bytes, nbytes);
    std::array<uint8_t, SHA256_DIGEST_SIZE> out{};
    sha256_final(&ctx, out.data());
    return out;
}

static bool cb_commit(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cd = (vi_trace_commit_data *) user_data;
    if (ask) {
        return tensor_name_matches(t->name, cd->filters);
    }
    if (!tensor_name_matches(t->name, cd->filters)) {
        return true;
    }

    const bool      is_host = ggml_backend_buffer_is_host(t->buffer);
    const size_t    nbytes  = ggml_nbytes(t);
    const uint8_t * data    = nullptr;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        cd->scratch.resize(nbytes);
        ggml_backend_tensor_get(t, cd->scratch.data(), 0, nbytes);
        data = cd->scratch.data();
    }

    trace_entry_meta m;
    m.name      = t->name;
    m.op_desc   = ggml_op_desc(t);
    m.type_name = ggml_type_name(t->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        m.ne[i] = t->ne[i];
    }
    m.nbytes = nbytes;

    cd->meta.emplace_back(std::move(m));
    cd->leaf_hashes.emplace_back(hash_trace_leaf(cd->meta.back(), data, nbytes));

    return true;
}

struct opening_record {
    size_t                                               leaf_index = 0;
    size_t                                               name_occurrence = 0;  // nth time this tensor name appears in the trace
    trace_entry_meta                                     meta;
    std::array<uint8_t, SHA256_DIGEST_SIZE>              leaf_hash{};
    std::vector<std::array<uint8_t, SHA256_DIGEST_SIZE>> merkle_path;
    std::filesystem::path                                bytes_path;
    std::array<uint8_t, SHA256_DIGEST_SIZE>              bytes_hash{};
};

static bool verify_openings(
    const std::array<uint8_t, SHA256_DIGEST_SIZE> & merkle_root,
    const std::vector<opening_record> & openings) {
    size_t ok = 0;

    for (size_t i = 0; i < openings.size(); ++i) {
        const auto & o = openings[i];

        // 1) Re-hash the opened bytes and check it matches the recorded hash.
        std::ifstream f(o.bytes_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "verify: opening[%zu]: cannot read bytes file: %s\n", i, o.bytes_path.string().c_str());
            continue;
        }
        std::vector<uint8_t> bytes((size_t) o.meta.nbytes);
        f.read((char *) bytes.data(), (std::streamsize) bytes.size());
        if ((size_t) f.gcount() != bytes.size()) {
            std::fprintf(stderr,
                         "verify: opening[%zu]: short read (%zu/%zu) from %s\n",
                         i,
                         (size_t) f.gcount(),
                         bytes.size(),
                         o.bytes_path.string().c_str());
            continue;
        }

        const auto bytes_hash = sha256_bytes(bytes.data(), bytes.size());
        if (bytes_hash != o.bytes_hash) {
            std::fprintf(stderr, "verify: opening[%zu]: bytes sha256 mismatch\n", i);
            continue;
        }

        // 2) Recompute the committed leaf hash from meta + bytes.
        const auto leaf_hash = hash_trace_leaf(o.meta, bytes.data(), bytes.size());
        if (leaf_hash != o.leaf_hash) {
            std::fprintf(stderr, "verify: opening[%zu]: leaf sha256 mismatch (meta/bytes)\n", i);
            continue;
        }

        // 3) Recompute the Merkle root from the leaf + sibling path.
        const auto root2 = merkle_recompute_root(o.leaf_index, leaf_hash, o.merkle_path);
        if (root2 != merkle_root) {
            std::fprintf(stderr, "verify: opening[%zu]: merkle root mismatch\n", i);
            continue;
        }

        ok++;
    }

    std::printf("verify: %zu/%zu openings verified against merkle root\n", ok, openings.size());
    return ok == openings.size();
}

struct vi_trace_open_data {
    // map tensor name -> (occurrence index -> opening index)
    std::unordered_map<std::string, std::unordered_map<size_t, size_t>> want;
    std::unordered_map<std::string, size_t>                              seen;
    std::unordered_map<const ggml_tensor *, size_t>                       armed;
    std::vector<uint8_t>                    scratch;

    std::vector<opening_record> * openings = nullptr;
};

static bool cb_open(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * od = (vi_trace_open_data *) user_data;
    if (ask) {
        const std::string name = t->name;
        const size_t ord = od->seen[name]++;

        auto it_name = od->want.find(name);
        if (it_name == od->want.end()) {
            return false;
        }

        auto it_ord = it_name->second.find(ord);
        if (it_ord == it_name->second.end()) {
            return false;
        }

        // Arm this specific tensor instance for the follow-up ask=false callback.
        od->armed[t] = it_ord->second;
        return true;
    }

    auto it_arm = od->armed.find(t);
    if (it_arm == od->armed.end()) {
        return true;
    }
    const size_t opening_idx = it_arm->second;
    od->armed.erase(it_arm);

    const bool      is_host = ggml_backend_buffer_is_host(t->buffer);
    const size_t    nbytes  = ggml_nbytes(t);
    const uint8_t * data    = nullptr;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        od->scratch.resize(nbytes);
        ggml_backend_tensor_get(t, od->scratch.data(), 0, nbytes);
        data = od->scratch.data();
    }

    auto & rec     = (*od->openings)[opening_idx];
    rec.bytes_hash = sha256_bytes(data, nbytes);

    std::ofstream f(rec.bytes_path, std::ios::binary);
    f.write((const char *) data, (std::streamsize) nbytes);
    f.close();

    return true;
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const bool add_special   = true;
    const bool parse_special = true;

    // llama_tokenize returns a negative number on failure:
    //   -number_of_tokens_that_would_have_been_returned
    const int n_req = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, add_special, parse_special);
    if (n_req == 0 || n_req == INT32_MIN) {
        return {};
    }
    const int n = n_req < 0 ? -n_req : n_req;

    std::vector<llama_token> toks((size_t) n);
    const int n2 = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(),
                                  add_special, parse_special);
    if (n2 < 0) {
        return {};
    }
    toks.resize((size_t) n2);
    return toks;
}

static bool run_decode_once(llama_context * ctx, const llama_vocab * vocab, const std::string & prompt) {
    const auto toks = tokenize_prompt(vocab, prompt);
    if (toks.empty()) {
        std::fprintf(stderr, "error: failed to tokenize prompt\n");
        return false;
    }

    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (int i = 0; i < (int) toks.size(); ++i) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (i == (int) toks.size() - 1);
    }
    batch.n_tokens = (int) toks.size();

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::fprintf(stderr, "error: llama_decode failed (%d)\n", rc);
        return false;
    }
    return true;
}

static std::array<uint8_t, SHA256_DIGEST_SIZE> derive_challenge_seed(
    const std::array<uint8_t, SHA256_DIGEST_SIZE> & merkle_root,
    uint32_t                                        user_seed) {
    uint8_t buf[SHA256_DIGEST_SIZE + sizeof(user_seed)];
    memcpy(buf, merkle_root.data(), SHA256_DIGEST_SIZE);
    memcpy(buf + SHA256_DIGEST_SIZE, &user_seed, sizeof(user_seed));
    return sha256_bytes(buf, sizeof(buf));
}

static std::vector<size_t> sample_indices(size_t n, int32_t k, const std::array<uint8_t, SHA256_DIGEST_SIZE> & seed) {
    if (n == 0 || k <= 0) {
        return {};
    }
    const size_t kk = (size_t) std::min<int32_t>(k, (int32_t) n);

    // Use 64-bit seed from first 8 bytes (fine for sampling demo).
    uint64_t s = 0;
    memcpy(&s, seed.data(), sizeof(s));
    std::mt19937_64 rng(s);

    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) {
        idx[i] = i;
    }
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(kk);
    std::sort(idx.begin(), idx.end());
    return idx;
}

static void write_text_file(const std::filesystem::path & p, const std::string & s) {
    std::ofstream f(p);
    f << s;
    f.close();
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    vi_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // Optional: download model via Hugging Face cache.
    // Mirrors the behavior used elsewhere in the repo: if hf_repo is set, resolve a GGUF file
    // (either hf_file explicitly, or by tag-based selection in common_download_model).
    if (!params.hf_repo.empty()) {
        common_params_model m;
        m.hf_repo = params.hf_repo;
        m.hf_file = params.hf_file;

        common_download_model_opts opts;
        opts.offline = params.offline;

        const auto res = common_download_model(m, params.hf_token, opts);
        if (res.model_path.empty()) {
            std::fprintf(stderr, "error: failed to download/resolve model from HF repo '%s'\n", params.hf_repo.c_str());
            return 1;
        }
        params.model_path = res.model_path;
    }

    std::filesystem::create_directories(params.out_dir);
    std::filesystem::create_directories(std::filesystem::path(params.out_dir) / "openings");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "error: failed to load model: %s\n", params.model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // -------------------------
    // Pass 1: commit to trace
    // -------------------------
    vi_trace_commit_data commit_data;
    for (const auto & pat : params.tensor_filters) {
        try {
            commit_data.filters.emplace_back("^" + pat, std::regex::optimize);
        } catch (const std::regex_error & e) {
            std::fprintf(stderr, "error: invalid regex '%s': %s\n", pat.c_str(), e.what());
            return 1;
        }
    }

    llama_context_params cparams1 = llama_context_default_params();
    cparams1.cb_eval              = cb_commit;
    cparams1.cb_eval_user_data    = &commit_data;

    llama_context * ctx1 = llama_new_context_with_model(model, cparams1);
    if (!ctx1) {
        std::fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    if (!run_decode_once(ctx1, vocab, params.prompt)) {
        llama_free(ctx1);
        llama_model_free(model);
        return 1;
    }

    if (commit_data.leaf_hashes.empty()) {
        std::fprintf(stderr, "error: empty trace (check --vi-tensor-filter)\n");
        llama_free(ctx1);
        llama_model_free(model);
        return 1;
    }

    merkle_tree mt       = build_merkle(commit_data.leaf_hashes);
    const auto  root     = mt.root();
    const auto  root_hex = hex_encode(root.data(), root.size());

    llama_free(ctx1);

    // Derive “challenge” from root (Fiat–Shamir style) and sample indices.
    const auto chal_seed = derive_challenge_seed(root, params.seed);
    const auto sampled   = sample_indices(commit_data.meta.size(), params.n_samples, chal_seed);

    // Write commit.json (minimal, human-readable JSON).
    {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"paper\": \"paper/2026-541.pdf\",\n";
        oss << "  \"scheme\": \"commit-and-open over ggml node outputs (demo)\",\n";
        oss << "  \"model\": " << std::quoted(params.model_path) << ",\n";
        oss << "  \"prompt\": " << std::quoted(params.prompt) << ",\n";
        oss << "  \"trace_leaves\": " << commit_data.meta.size() << ",\n";
        oss << "  \"merkle_root_sha256\": " << std::quoted(root_hex) << ",\n";
        oss << "  \"challenge_seed_sha256\": " << std::quoted(hex_encode(chal_seed.data(), chal_seed.size())) << ",\n";
        oss << "  \"openings\": {\n";
        oss << "    \"n_samples\": " << sampled.size() << ",\n";
        oss << "    \"indices\": [";
        for (size_t i = 0; i < sampled.size(); ++i) {
            oss << sampled[i] << (i + 1 < sampled.size() ? ", " : "");
        }
        oss << "]\n";
        oss << "  }\n";
        oss << "}\n";
        write_text_file(std::filesystem::path(params.out_dir) / "commit.json", oss.str());
    }

    // -------------------------
    // Pass 2: open sampled entries (re-run and capture bytes)
    // -------------------------
    std::vector<opening_record> openings;
    openings.resize(sampled.size());

    vi_trace_open_data open_data;
    open_data.openings = &openings;

    // Compute per-entry occurrence index for each tensor name in the committed trace.
    std::vector<size_t> name_occ(commit_data.meta.size());
    {
        std::unordered_map<std::string, size_t> occ;
        for (size_t i = 0; i < commit_data.meta.size(); ++i) {
            name_occ[i] = occ[commit_data.meta[i].name]++;
        }
    }

    // Build lookup by (tensor name, occurrence index) for sampled openings.
    for (size_t j = 0; j < sampled.size(); ++j) {
        const size_t leaf_idx = sampled[j];
        const auto & meta     = commit_data.meta[leaf_idx];

        opening_record rec;
        rec.leaf_index  = leaf_idx;
        rec.name_occurrence = name_occ[leaf_idx];
        rec.meta        = meta;
        rec.leaf_hash   = commit_data.leaf_hashes[leaf_idx];
        rec.merkle_path = merkle_proof(mt, leaf_idx);
        rec.bytes_path = std::filesystem::path(params.out_dir) / "openings" /
            (std::to_string(leaf_idx) + "-" + safe_file_component(meta.name) + ".bin");

        openings[j] = std::move(rec);
        open_data.want[meta.name].emplace(name_occ[leaf_idx], j);
    }

    llama_context_params cparams2 = llama_context_default_params();
    cparams2.cb_eval              = cb_open;
    cparams2.cb_eval_user_data    = &open_data;

    llama_context * ctx2 = llama_new_context_with_model(model, cparams2);
    if (!ctx2) {
        std::fprintf(stderr, "error: failed to create context (pass2)\n");
        llama_model_free(model);
        return 1;
    }

    if (!run_decode_once(ctx2, vocab, params.prompt)) {
        llama_free(ctx2);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx2);
    llama_model_free(model);
    llama_backend_free();

    // Write openings.json.
    {
        std::ostringstream oss;
        oss << "{\n";
        oss << "  \"merkle_root_sha256\": " << std::quoted(root_hex) << ",\n";
        oss << "  \"openings\": [\n";
        for (size_t i = 0; i < openings.size(); ++i) {
            const auto & o = openings[i];
            oss << "    {\n";
            oss << "      \"leaf_index\": " << o.leaf_index << ",\n";
            oss << "      \"name_occurrence\": " << o.name_occurrence << ",\n";
            oss << "      \"tensor\": {\n";
            oss << "        \"name\": " << std::quoted(o.meta.name) << ",\n";
            oss << "        \"op\": " << std::quoted(o.meta.op_desc) << ",\n";
            oss << "        \"type\": " << std::quoted(o.meta.type_name) << ",\n";
            oss << "        \"ne\": [" << o.meta.ne[0] << ", " << o.meta.ne[1] << ", " << o.meta.ne[2] << ", "
                << o.meta.ne[3] << "],\n";
            oss << "        \"nbytes\": " << o.meta.nbytes << "\n";
            oss << "      },\n";
            oss << "      \"leaf_hash_sha256\": " << std::quoted(hex_encode(o.leaf_hash.data(), o.leaf_hash.size()))
                << ",\n";
            oss << "      \"opened_bytes\": {\n";
            oss << "        \"path\": " << std::quoted(o.bytes_path.string()) << ",\n";
            oss << "        \"sha256\": " << std::quoted(hex_encode(o.bytes_hash.data(), o.bytes_hash.size())) << "\n";
            oss << "      },\n";
            oss << "      \"merkle_path_siblings_sha256\": [";
            for (size_t k = 0; k < o.merkle_path.size(); ++k) {
                oss << std::quoted(hex_encode(o.merkle_path[k].data(), o.merkle_path[k].size()));
                oss << (k + 1 < o.merkle_path.size() ? ", " : "");
            }
            oss << "]\n";
            oss << "    }" << (i + 1 < openings.size() ? "," : "") << "\n";
        }
        oss << "  ]\n";
        oss << "}\n";
        write_text_file(std::filesystem::path(params.out_dir) / "openings.json", oss.str());
    }

    std::printf("wrote %s\n", (std::filesystem::path(params.out_dir) / "commit.json").string().c_str());
    std::printf("wrote %s\n", (std::filesystem::path(params.out_dir) / "openings.json").string().c_str());
    std::printf("merkle_root_sha256=%s\n", root_hex.c_str());

    (void) verify_openings(root, openings);

    return 0;
}
