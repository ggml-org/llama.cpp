#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
using Prob = std::vector<double>;

static void check(bool ok, const std::string & msg) {
    if (!ok) throw std::runtime_error(msg);
}

static double uniform(std::mt19937_64 & rng) {
    return std::generate_canonical<double, 53>(rng);
}

// Temperature precedes nucleus truncation. No top-k, min-p or penalties.
static Prob distribution(const float * logits, int n, bool greedy = false) {
    const int best = std::max_element(logits, logits + n) - logits;
    Prob p(n, 0.0);
    if (greedy) { p[best] = 1.0; return p; }
    double total = 0;
    for (int i = 0; i < n; ++i) {
        check(!std::isnan(logits[i]), "NaN logit");
        p[i] = std::exp(double(logits[i]) - logits[best]);
        total += p[i];
    }
    check(total > 0 && std::isfinite(total), "invalid probability mass");
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    auto cmp = [&](int a, int b) { return p[a] == p[b] ? a < b : p[a] > p[b]; };
    int count = std::min(n, 64), keep = 0;
    double mass = 0;
    while (true) {
        std::partial_sort(order.begin(), order.begin() + count, order.end(), cmp);
        mass = 0;
        for (keep = 0; keep < count; ++keep) {
            mass += p[order[keep]];
            if (mass >= 0.95 * total) { ++keep; break; }
        }
        if (mass >= 0.95 * total || count == n) break;
        count = std::min(n, count * 4);
    }
    Prob out(n, 0);
    for (int j = 0; j < keep; ++j) out[order[j]] = p[order[j]] / mass;
    return out;
}

static int sample(const Prob & p, std::mt19937_64 & rng) {
    double u = uniform(rng), sum = 0;
    int last = -1;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] <= 0) continue;
        last = int(i);
        sum += p[i];
        if (u < sum) return int(i);
    }
    check(last >= 0 && std::abs(sum - 1) < 1e-8, "invalid categorical distribution");
    return last;
}

static double acceptance(const Prob & p, const Prob & q, int y) {
    check(q[y] > 0, "sample outside draft support");
    return std::min(1.0, p[y] / q[y]);
}

static Prob residual(const Prob & p, const Prob & q) {
    Prob r(p.size());
    double mass = 0;
    for (size_t i = 0; i < p.size(); ++i) { r[i] = std::max(0.0, p[i] - q[i]); mass += r[i]; }
    check(mass > 0, "rejection with zero residual mass");
    for (double & v : r) v /= mass;
    return r;
}

static void self_test() {
    const std::vector<std::pair<Prob, Prob>> cases = {
        {{.1, .3, .6}, {.6, .3, .1}}, {{.2, .5, .3}, {.2, .5, .3}},
        {{1, 0, 0}, {0, .7, .3}}, {{.1, .3, .6}, {0, 1, 0}},
    };
    for (const auto & c : cases) {
        std::mt19937_64 rq(1), ra(2), rp(3);
        std::vector<int> hist(3);
        int accepted = 0;
        constexpr int trials = 250000;
        for (int j = 0; j < trials; ++j) {
            int y = sample(c.second, rq);
            bool a = uniform(ra) < acceptance(c.first, c.second, y);
            accepted += a;
            hist[a ? y : sample(residual(c.first, c.second), rp)]++;
        }
        double expected = 0;
        for (int i = 0; i < 3; ++i) {
            check(std::abs(double(hist[i]) / trials - c.first[i]) < .005, "marginal distribution self-test failed");
            expected += std::min(c.first[i], c.second[i]);
        }
        check(std::abs(double(accepted) / trials - expected) < .005, "acceptance self-test failed");
    }
    float l[] = {float(std::log(.6)), float(std::log(.3)), float(std::log(.1))};
    auto p = distribution(l, 3);
    check(p[2] > 0, "nucleus must include crossing token");
    float l2[] = {0, -1, -10};
    p = distribution(l2, 3);
    check(p[2] == 0 && std::abs(p[0] + p[1] - 1) < 1e-12, "nucleus truncation failed");
    std::cout << "PASS: maximal coupling marginals/acceptance (4 x 250000), nucleus crossing, truncation\n";
}

// These graphs reuse llama.cpp's already resident weights on the same CUDA device.
struct HeadGraph {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    HeadGraph(ggml_backend_t backend, ggml_tensor * weight, ggml_tensor * w1, int width) {
        ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        check(ctx, "head metadata allocation");
        graph = ggml_new_graph_custom(ctx, 128, false);
        input = w1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1)
                   : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, weight->ne[0], width);
        ggml_set_input(input);
        auto * x = w1 ? ggml_get_rows(ctx, w1, input) : input;
        output = ggml_mul_mat(ctx, weight, x);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        check(ggml_gallocr_alloc_graph(alloc, graph), "head graph allocation");
    }
    ~HeadGraph() { ggml_gallocr_free(alloc); ggml_free(ctx); }
    std::vector<float> compute(ggml_backend_t backend, const void * data, size_t bytes) {
        ggml_backend_tensor_set(input, data, 0, bytes);
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "head graph compute");
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};

struct Batch {
    llama_batch b;
    Batch(int n, int embd = 0) : b(llama_batch_init(n, embd, 1)) {
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
        }
    }
    ~Batch() { llama_batch_free(b); }
};

struct Engine {
    llama_model * target = nullptr;
    llama_model * draft = nullptr;
    llama_context * tc = nullptr;
    llama_context * dc = nullptr;
    ggml_backend_t backend = nullptr;
    std::unique_ptr<HeadGraph> bases[8];
    std::unique_ptr<HeadGraph> bias;
    const llama_vocab * vocab = nullptr;
    std::vector<int32_t> layers;
    int dim, nv, mask;
    bool greedy;
    bool head_checked = false;
    std::string dump;

    Engine(const std::string & target_path, const std::string & draft_path, bool greedy_, std::string dump_)
        : greedy(greedy_), dump(std::move(dump_)) {
        ggml_backend_load_all();
        llama_backend_init();
        auto dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        check(dev, "no GPU backend loaded");
        std::cerr << "GPU: " << ggml_backend_dev_description(dev) << '\n';
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        target = llama_model_load_from_file(target_path.c_str(), mp);
        check(target, "target model load");
        draft = llama_model_load_from_file(draft_path.c_str(), mp);
        check(draft, "draft model load");
        dim = llama_model_n_embd(target);
        vocab = llama_model_get_vocab(target);
        nv = llama_vocab_n_tokens(vocab);
        mask = llama_vocab_mask(llama_model_get_vocab(draft));
        check(dim == 2560 && nv == 131072 && mask >= 0, "unexpected model dimensions or mask token");
        check(target->output && target->output->ne[0] == dim && target->output->ne[1] == nv,
              "target output tensor ABI/shape mismatch");
        check(draft->dspark_markov_w1 && draft->dspark_markov_w2 && !draft->d2t,
              "requires vanilla Markov head and full vocabulary");
        check(draft->dspark_markov_w1->ne[0] == 256 && draft->dspark_markov_w1->ne[1] == nv,
              "Markov tensor ABI/shape mismatch");
        auto cp = llama_context_default_params();
        cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 128; cp.n_seq_max = 1;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 8;
        cp.swa_full = false;
        cp.n_threads = cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        tc = llama_init_from_model(target, cp);
        check(tc, "target context");
        cp.ctx_other = tc; cp.embeddings = false; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
        dc = llama_init_from_model(draft, cp);
        check(dc, "draft context");
        auto ids = llama_model_target_layer_ids(draft);
        layers.assign(ids, ids + llama_model_target_layer_ids_n(draft));
        check(layers == std::vector<int32_t>({3, 11, 19, 26, 34}), "unexpected target feature layers");
        for (int lid : layers) llama_set_embeddings_layer_inp(tc, lid, true);
        llama_set_embeddings_nextn(dc, true, true);
        llama_set_causal_attn(dc, false);
        backend = ggml_backend_dev_init(dev, nullptr);
        check(backend, "head backend");
        bias.reset(new HeadGraph(backend, draft->dspark_markov_w2, draft->dspark_markov_w1, 1));
    }

    ~Engine() {
        bias.reset(); for (auto & b : bases) b.reset();
        ggml_backend_free(backend);
        llama_free(dc); llama_free(tc);
        llama_model_free(draft); llama_model_free(target); llama_backend_free();
    }

    void decode(llama_context * c, Batch & b) {
        check(llama_decode(c, b.b) == 0, "llama_decode failed");
    }

    void inject(int n, int pos) {
        // The fusion encoder exposes nextn features, not regular pooled embeddings.
        llama_set_embeddings(dc, false);
        std::vector<float> features(size_t(n) * dim * layers.size());
        for (size_t k = 0; k < layers.size(); ++k) {
            const float * x = llama_get_embeddings_layer_inp(tc, layers[k]);
            check(x, "missing target features");
            for (int i = 0; i < n; ++i)
                std::memcpy(features.data() + (size_t(i) * layers.size() + k) * dim,
                            x + size_t(i) * dim, dim * sizeof(float));
        }
        llama_batch enc = {n, nullptr, features.data(), nullptr, nullptr, nullptr, nullptr};
        check(llama_encode(dc, enc) == 0, "feature fusion failed");
        auto fused = llama_get_embeddings_nextn(dc);
        check(fused, "missing fused features");
        Batch b(n, dim);
        std::memcpy(b.b.embd, fused, size_t(n) * dim * sizeof(float));
        for (int i = 0; i < n; ++i) { b.b.pos[i] = pos + i; b.b.logits[i] = false; }
        decode(dc, b);
    }

    std::vector<Prob> propose(int anchor, int pos, int width, std::mt19937_64 & rng,
                             std::vector<llama_token> & ys, double & parity_error) {
        llama_set_embeddings(dc, true);
        Batch b(width);
        for (int i = 0; i < width; ++i) {
            b.b.token[i] = i == 0 ? anchor : mask; b.b.pos[i] = pos + i;
        }
        decode(dc, b);
        std::vector<float> hidden(size_t(dim) * width);
        for (int i = 0; i < width; ++i) {
            auto * x = llama_get_embeddings_ith(dc, i);
            check(x, "missing draft backbone hidden state");
            std::memcpy(hidden.data() + size_t(i) * dim, x, dim * sizeof(float));
        }
        if (!bases[width]) bases[width].reset(new HeadGraph(backend, target->output, nullptr, width));
        auto base = bases[width]->compute(backend, hidden.data(), hidden.size() * sizeof(float));
        std::vector<Prob> qs;
        int prev = anchor;
        std::vector<float> logits(nv);
        const bool save_block = !head_checked && !dump.empty();
        std::vector<float> saved_bias;
        std::vector<int32_t> saved_prev;
        for (int i = 0; i < width; ++i) {
            auto bi = bias->compute(backend, &prev, sizeof(prev));
            if (save_block) {
                saved_prev.push_back(prev);
                saved_bias.insert(saved_bias.end(), bi.begin(), bi.end());
            }
            for (int j = 0; j < nv; ++j) logits[j] = base[size_t(i) * nv + j] + bi[j];
            if (greedy) {
                const float * native = llama_get_logits_ith(dc, i);
                check(std::max_element(native, native + nv) - native ==
                      std::max_element(logits.begin(), logits.end()) - logits.begin(),
                      "greedy external/native Markov chain top-1 mismatch");
            }
            if (!head_checked && i == 0) {
                // Slot zero has identical conditioning in the native greedy chain.
                const float * native = llama_get_logits_ith(dc, 0);
                double err = 0;
                for (int j = 0; j < nv; ++j) err = std::max(err, double(std::abs(native[j] - logits[j])));
                parity_error = err;
                std::cerr << "head slot-0 max absolute error vs native: " << err << '\n';
                check(err < .25, "external/native head parity failed");
                if (!dump.empty()) {
                    auto save = [&](const std::string & name, const std::vector<float> & v) {
                        std::ofstream f(dump + name, std::ios::binary);
                        check(bool(f), "cannot open head dump");
                        f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float));
                    };
                    save(".hidden.f32", hidden); save(".base.f32", base); save(".bias.f32", bi);
                }
                head_checked = true;
            }
            qs.push_back(distribution(logits.data(), nv, greedy));
            prev = sample(qs.back(), rng);
            ys.push_back(prev);
        }
        if (save_block) {
            std::ofstream f(dump + ".all_bias.f32", std::ios::binary);
            f.write(reinterpret_cast<const char *>(saved_bias.data()), saved_bias.size() * sizeof(float));
            std::ofstream p(dump + ".prev.i32", std::ios::binary);
            p.write(reinterpret_cast<const char *>(saved_prev.data()), saved_prev.size() * sizeof(int32_t));
            std::ofstream y(dump + ".sampled.i32", std::ios::binary);
            y.write(reinterpret_cast<const char *>(ys.data()), ys.size() * sizeof(llama_token));
        }
        return qs;
    }

    json run(const json & item, int max_tokens, uint64_t seed) {
        const auto started = Clock::now();
        auto prompt = item.at("prompt_tokens").get<std::vector<llama_token>>();
        check(!prompt.empty() && prompt.size() + max_tokens + 8 < 8192, "context exceeded");
        llama_memory_clear(llama_get_memory(tc), false);
        llama_memory_clear(llama_get_memory(dc), false);
        for (int off = 0; off < int(prompt.size()); off += 128) {
            const int n = std::min(128, int(prompt.size()) - off);
            Batch b(n);
            for (int i = 0; i < n; ++i) {
                b.b.token[i] = prompt[off + i]; b.b.pos[i] = off + i;
                b.b.logits[i] = i == n - 1;
            }
            decode(tc, b); inject(n, off);
        }
        std::mt19937_64 rq(seed ^ 0x91d168b7ULL), ra(seed ^ 0x5abe29d3ULL), rp(seed ^ 0xda21fc85ULL);
        std::vector<llama_token> output;
        int anchor = sample(distribution(llama_get_logits_ith(tc, -1), nv, greedy), rp);
        output.push_back(anchor);
        int pos = prompt.size(), accepted = 0, proposed = 0, verifies = 0;
        json rounds = json::array();
        bool eos = llama_vocab_is_eog(vocab, anchor);
        double parity_error = -1;
        while (int(output.size()) < max_tokens && !eos) {
            const int width = std::min(7, max_tokens - int(output.size()) - 1);
            if (width == 0) {
                Batch b(1); b.b.token[0] = anchor; b.b.pos[0] = pos;
                decode(tc, b);
                output.push_back(sample(distribution(llama_get_logits_ith(tc, 0), nv, greedy), rp));
                eos = llama_vocab_is_eog(vocab, output.back());
                break;
            }
            std::vector<llama_token> ys;
            auto qs = propose(anchor, pos, width, rq, ys, parity_error);
            Batch b(width + 1);
            b.b.token[0] = anchor;
            for (int i = 0; i <= width; ++i) {
                if (i > 0) b.b.token[i] = ys[i - 1];
                b.b.pos[i] = pos + i;
            }
            decode(tc, b);
            ++verifies; proposed += width;
            int k = 0, next = -1;
            json decisions = json::array();
            for (; k < width; ++k) {
                auto p = distribution(llama_get_logits_ith(tc, k), nv, greedy);
                double a = acceptance(p, qs[k], ys[k]);
                double u = uniform(ra);
                bool ok = u < a;
                decisions.push_back({{"token", ys[k]}, {"p", p[ys[k]]}, {"q", qs[k][ys[k]]},
                                     {"u", u}, {"accept", ok}});
                if (!ok) { next = sample(residual(p, qs[k]), rp); break; }
                output.push_back(ys[k]);
                ++accepted;
                if (llama_vocab_is_eog(vocab, ys[k])) { eos = true; ++k; break; }
            }
            if (!eos && k == width)
                next = sample(distribution(llama_get_logits_ith(tc, width), nv, greedy), rp);
            rounds.push_back({{"width", width}, {"accepted", k}, {"decisions", decisions}});
            if (eos) break;
            check(next >= 0, "missing correction/bonus");
            output.push_back(next); anchor = next;
            eos = llama_vocab_is_eog(vocab, anchor);
            // Only committed target rows seed the draft cache; noise rows never persist.
            check(llama_memory_seq_rm(llama_get_memory(dc), 0, pos, -1), "draft rollback failed");
            inject(k + 1, pos);
            pos += k + 1;
            check(llama_memory_seq_rm(llama_get_memory(tc), 0, pos, -1), "target rollback failed");
        }
        std::string text;
        for (int token : output) {
            std::vector<char> piece(256);
            int n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
            if (n < 0) { piece.resize(-n); n = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true); }
            check(n >= 0, "token detokenization failed");
            text.append(piece.data(), n);
        }
        return {{"id", item.at("id")}, {"dataset", item.at("dataset")}, {"seed", seed},
                {"prompt_tokens", prompt.size()}, {"generated_tokens", output.size()}, {"tokens", output},
                {"text", text}, {"finish_reason", eos ? "eog" : "length"},
                {"draft_tokens", proposed}, {"accepted_tokens", accepted}, {"verification_steps", verifies},
                {"tau", verifies ? 1 + double(accepted) / verifies : 1.0}, {"rounds", rounds},
                {"head_parity_max_abs", parity_error},
                {"seconds", std::chrono::duration<double>(Clock::now() - started).count()}};
    }
};

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") { self_test(); return 0; }
        check(argc >= 7, "usage: bench TARGET DRAFT CASES.jsonl RESULTS.jsonl MAX_TOKENS SEED [--greedy] [--dump=PREFIX]");
        bool greedy = false; std::string dump;
        for (int i = 7; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--greedy") greedy = true;
            else if (arg.rfind("--dump=", 0) == 0) dump = arg.substr(7);
            else throw std::runtime_error("unknown argument: " + arg);
        }
        int max_tokens = std::stoi(argv[5]); uint64_t seed = std::stoull(argv[6]);
        check(max_tokens >= 2 && max_tokens <= 2048, "invalid max_tokens");
        std::ifstream cases(argv[3]); check(bool(cases), "cannot open cases");
        std::set<std::string> completed;
        std::string line;
        { std::ifstream old(argv[4]); while (std::getline(old, line)) completed.insert(json::parse(line).at("id")); }
        std::ofstream out(argv[4], std::ios::app); check(bool(out), "cannot open results");
        Engine engine(argv[1], argv[2], greedy, dump);
        int count = completed.size();
        while (std::getline(cases, line)) {
            auto item = json::parse(line);
            if (completed.count(item.at("id").get<std::string>())) continue;
            auto result = engine.run(item, max_tokens, seed + item.at("index").get<uint64_t>());
            out << result.dump(-1, ' ', false, json::error_handler_t::replace) << '\n'; out.flush();
            std::cerr << "DONE " << ++count << " " << item.at("id") << " generated=" << result["generated_tokens"]
                      << " tau=" << result["tau"] << " seconds=" << result["seconds"] << '\n';
        }
        return 0;
    } catch (const std::exception & e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
