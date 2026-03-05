#ifdef GGML_COMMON_DECL
#undef GGML_COMMON_DECL
#endif
#define GGML_COMMON_DECL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-impl.h"

#include "llama-quant.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "../ggml/include/ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <regex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../ggml/src/ggml-nvfp4-helpers.h"

#ifndef NVFP4_A0
#define NVFP4_A0 0.9918823242f
#endif
#ifndef NVFP4_B0
#define NVFP4_B0 0.9864501953f
#endif

// Quantization types. Changes to this struct must be replicated in quantize.cpp
struct tensor_quantization {
    std::string name;
    ggml_type quant = GGML_TYPE_COUNT;
};

static int64_t llama_nvfp4_autotune_sample_blocks(const int64_t nb_total) {
    if (nb_total <= 0) {
        return 0;
    }
    // Keep a larger host sample pool so CUDA adaptive-resample can escalate
    // from the 256-block initial pass to 512/1024 when tensors are risky.
    const int64_t cap =
        nb_total >= 16384 ? 1024 :
        nb_total >= 8192  ? 512  :
        nb_total >= 2048  ? 256  : 128;
    return std::min(nb_total, cap);
}

static inline int64_t llama_nvfp4_sample_block_index(const int64_t is, const int64_t sample_nb, const int64_t nb_total) {
    if (nb_total <= 1 || sample_nb <= 1) {
        return 0;
    }
    // Deterministic uniform coverage across full tensor block range [0, nb_total - 1].
    return (is * (nb_total - 1)) / (sample_nb - 1);
}

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static inline float llama_tensor_absmax(const float * data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float v = std::fabs(data[i]);
        if (v > max_abs) max_abs = v;
    }
    return max_abs;
}

static float llama_tensor_absmax_f32_mt(const float * data, size_t n, int nthread) {
    if (nthread < 2 || n < 1024) {
        return llama_tensor_absmax(data, n);
    }

    const int nthreads = std::max(1, nthread);
    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);
    std::vector<float> partials((size_t) nthreads, 0.0f);

    auto worker = [&](int tid, size_t start, size_t end) {
        float m = 0.0f;
        for (size_t i = start; i < end; ++i) {
            const float v = std::fabs(data[i]);
            if (v > m) m = v;
        }
        partials[(size_t) tid] = m;
    };

    const size_t chunk = (n + (size_t) nthreads - 1) / (size_t) nthreads;
    for (int t = 0; t < nthreads - 1; ++t) {
        const size_t start = (size_t) t * chunk;
        const size_t end = std::min(n, start + chunk);
        threads.emplace_back(worker, t, start, end);
    }
    {
        const size_t start = (size_t) (nthreads - 1) * chunk;
        const size_t end = std::min(n, start + chunk);
        worker(nthreads - 1, start, end);
    }

    for (auto & th : threads) th.join();

    float max_abs = 0.0f;
    for (int t = 0; t < nthreads; ++t) {
        if (partials[(size_t) t] > max_abs) max_abs = partials[(size_t) t];
    }
    return max_abs;
}

static float llama_tensor_absmax_bf16_mt(const ggml_bf16_t * data, size_t n, int nthread) {
    if (nthread < 2 || n < 1024) {
        float max_abs = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float v = std::fabs(ggml_bf16_to_fp32(data[i]));
            if (v > max_abs) max_abs = v;
        }
        return max_abs;
    }

    const int nthreads = std::max(1, nthread);
    std::vector<std::thread> threads;
    threads.reserve(nthreads - 1);
    std::vector<float> partials((size_t) nthreads, 0.0f);

    auto worker = [&](int tid, size_t start, size_t end) {
        float m = 0.0f;
        for (size_t i = start; i < end; ++i) {
            const float v = std::fabs(ggml_bf16_to_fp32(data[i]));
            if (v > m) m = v;
        }
        partials[(size_t) tid] = m;
    };

    const size_t chunk = (n + (size_t) nthreads - 1) / (size_t) nthreads;
    for (int t = 0; t < nthreads - 1; ++t) {
        const size_t start = (size_t) t * chunk;
        const size_t end = std::min(n, start + chunk);
        threads.emplace_back(worker, t, start, end);
    }
    {
        const size_t start = (size_t) (nthreads - 1) * chunk;
        const size_t end = std::min(n, start + chunk);
        worker(nthreads - 1, start, end);
    }

    for (auto & th : threads) th.join();

    float max_abs = 0.0f;
    for (int t = 0; t < nthreads; ++t) {
        if (partials[(size_t) t] > max_abs) max_abs = partials[(size_t) t];
    }
    return max_abs;
}

static constexpr float LLAMA_NVFP4_MAX_FP4 = 6.0f;
static constexpr float LLAMA_NVFP4_TENSOR_CAP_FIXED = 448.0f;

// Exact UE4M3 decode (QA path; supports subnormals).
static inline float nvfp4_qa_fp8_to_fp32(uint8_t b) {
    const uint32_t u = b & 0x7Fu;
    if (u == 0) return 0.0f;

    const uint32_t exp  = u >> 3;
    const uint32_t mant = u & 0x7u;

    if (exp == 0xFu && mant == 0x7u) return 448.0f;

    uint32_t bits;
    if (exp != 0) {
        bits = ((exp + 120u) << 23) | (mant << 20);
    } else {
        const uint32_t p = 31u - (uint32_t)__builtin_clz(mant);
        const uint32_t r = mant - (1u << p);
        const uint32_t exp32  = (p + 118u);
        const uint32_t mant32 = r << (23u - p);
        bits = (exp32 << 23) | mant32;
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

// Quantize sb_ideal -> UE4M3 byte, then bump upward if needed to avoid FP4 clipping after rounding.
// invM = 1/6 or 1/4.
struct nvfp4_qa_stats {
    uint64_t n16 = 0;
    uint64_t choose4 = 0;
    uint64_t choose6 = 0;
    uint64_t clip6_m4 = 0;
    uint64_t clip6_m6 = 0;
    uint64_t round_down = 0;

    double sum_mse = 0.0;
    double sum_sq  = 0.0;
    float  max_err = 0.0f;

    uint64_t b0 = 0;
    uint64_t bsub = 0;     // 0x01..0x07
    uint64_t bminnorm = 0; // 0x08
    uint64_t bmax = 0;     // 0x7E

    uint64_t hist[256] = {};
};

struct nvfp4_qa_context {
    FILE * file = nullptr;
    bool q4k = false;
    mutable std::mutex mutex;
};

struct nvfp4_qa_tensor_ctx {
    const nvfp4_qa_context * qa = nullptr;
    const char * tensor_name = nullptr;
    std::atomic<int64_t> * chunk_id = nullptr;
    int64_t n_per_row = 0;
};

static void nvfp4_qa_accum(
    nvfp4_qa_stats & st,
    const float * GGML_RESTRICT x,
    int64_t nrows,
    int64_t n_per_row,
    const void * GGML_RESTRICT qdata
) {
    const int64_t nblocks = n_per_row / QK_K;
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, n_per_row);
    const auto * qbytes = (const uint8_t *) qdata;

    for (int64_t r = 0; r < nrows; ++r) {
        const float * xrow = x + r * n_per_row;
        const block_nvfp4 * row_packs = (const block_nvfp4 *)(qbytes + r * row_size);

        for (int64_t b = 0; b < nblocks; ++b) {
            const int64_t pack = b >> 2;
            const int lane = (int) (b & 3);
            const block_nvfp4 * p = &row_packs[pack];
            const float pack_scale = 1.0f;
            const uint8_t * scales = p->scales[lane];

            for (int sb = 0; sb < (QK_K / QK_NVFP4); ++sb) {
                const float * x16 = xrow + b * QK_K + sb * QK_NVFP4;

                float max_abs = 0.0f;
                for (int k = 0; k < QK_NVFP4; ++k) {
                    const float a = std::fabs(x16[k]);
                    if (a > max_abs) max_abs = a;
                }

                const uint8_t bscale = scales[sb];
                st.hist[bscale]++;

                if (bscale == 0x00) st.b0++;
                else if (bscale <= 0x07) st.bsub++;
                else if (bscale == 0x08) st.bminnorm++;
                else if (bscale == 0x7E) st.bmax++;

                const float sb_dec = nvfp4_qa_fp8_to_fp32(bscale);
                const float scale = pack_scale * sb_dec;

                const float ideal6 = (max_abs > 0.0f) ? (max_abs * (1.0f / 6.0f)) : 0.0f;
                const float ideal4 = (max_abs > 0.0f) ? (max_abs * (1.0f / 4.0f)) : 0.0f;

                const float e6 = std::fabs(scale - ideal6);
                const float e4 = std::fabs(scale - ideal4);
                if (e4 < e6) st.choose4++;
                else         st.choose6++;

                if (pack_scale > 0.0f && max_abs > 0.0f) {
                    const float sb_ideal6 = ideal6 / pack_scale;
                    const float sb_ideal4 = ideal4 / pack_scale;
                    const float sb_ideal = (e4 < e6) ? sb_ideal4 : sb_ideal6;
                    if (sb_dec < sb_ideal) st.round_down++;
                }

                if (max_abs > 0.0f && scale > 0.0f) {
                    float sub_mse = 0.0f;
                    for (int k = 0; k < QK_NVFP4; ++k) {
                        const float v   = x16[k];
                        const uint8_t qi = best_index_nvfp4(v, scale);
                        const float deq = kvalues_nvfp4_float(qi & 0xF) * scale;
                        const float err = deq - v;
                        const float ae  = std::fabs(err);
                        sub_mse += err * err;
                        if (ae > st.max_err) st.max_err = ae;
                        st.sum_sq += (double) v * v;
                    }
                    st.sum_mse += sub_mse;

                    if (max_abs > LLAMA_NVFP4_MAX_FP4 * scale) {
                        if (e4 < e6) st.clip6_m4++;
                        else         st.clip6_m6++;
                    }
                }

                st.n16++;
            }
        }
    }
}

static std::string nvfp4_qa_hist_to_string(const nvfp4_qa_stats & st) {
    std::string out;
    char buf[64];
    for (int b = 0; b < 256; ++b) {
        const uint64_t count = st.hist[b];
        if (!count) continue;
        if (!out.empty()) out.push_back(',');
        std::snprintf(buf, sizeof(buf), "%02x:%llu", b, (unsigned long long) count);
        out.append(buf);
    }
    return out;
}

static void nvfp4_qa_write_line(
    const nvfp4_qa_context & qa,
    const char * tensor_name,
    int64_t chunk_id,
    const nvfp4_qa_stats & st
) {
    if (!qa.file) return;

    const double n16 = (double) st.n16;
    const double clip4_rate = n16 ? (100.0 * (double) st.clip6_m4 / n16) : 0.0;
    const double clip6_rate = n16 ? (100.0 * (double) st.clip6_m6 / n16) : 0.0;
    const double down_rate  = n16 ? (100.0 * (double) st.round_down / n16) : 0.0;
    const double c4_rate    = n16 ? (100.0 * (double) st.choose4 / n16) : 0.0;

    const double mse = n16 ? (st.sum_mse / (n16 * QK_NVFP4)) : 0.0;
    const double snr = (st.sum_mse > 0.0) ? (10.0 * std::log10(st.sum_sq / st.sum_mse)) : 99.0;

    const std::string hist = nvfp4_qa_hist_to_string(st);

    std::lock_guard<std::mutex> lock(qa.mutex);
    std::fprintf(qa.file,
        "%s\t%lld\t%llu\t%.8e\t%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%llu\t%llu\t%llu\t%llu\t%s\n",
        tensor_name,
        (long long) chunk_id,
        (unsigned long long) st.n16,
        mse,
        snr,
        (double) st.max_err,
        clip4_rate,
        clip6_rate,
        down_rate,
        c4_rate,
        (unsigned long long) st.bsub,
        (unsigned long long) st.bminnorm,
        (unsigned long long) st.bmax,
        (unsigned long long) st.b0,
        hist.c_str());
    std::fflush(qa.file);
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix (const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                LLAMA_LOG_DEBUG("(blk.%d imatrix) ", p.first);
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_k_quantized = 0;
    int n_fallback    = 0;

    bool has_imatrix = false;

    // used to figure out if a model shares tok_embd with the output weight
    bool has_output = false;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params)
        : model(model)
        , params(params)
        {}
};

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

//
static constexpr size_t LLAMA_QUANT_MIN_SAVINGS_BYTES = 4u * 1024u * 1024u;

static ggml_type llama_tensor_get_type(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;
    const auto       tn = LLM_TN(arch);

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (name == tn(LLM_TENSOR_OUTPUT, "weight") || (!qs.has_output && name == tn(LLM_TENSOR_TOKEN_EMBD, "weight"))) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);
            if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                       ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
                       ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ||
                       ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            } else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE ) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (name == "token_embd.weight" || name == "per_layer_token_embd.weight") {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } /* else if (ftype == LLAMA_FTYPE_MOSTLY_NVFP4) {
        new_type = GGML_TYPE_NVFP4;
        if (name.find("attn_v.weight") != std::string::npos) {
            if (use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) {
                new_type = GGML_TYPE_Q6_K;
            }
            ++qs.i_attention_wv;
        } else if (name.find("ffn_down") != std::string::npos) {
            auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
            int i_layer = info.first, n_layer = info.second;
            if (use_more_bits(i_layer, n_layer)) {
                new_type = GGML_TYPE_Q6_K;
            }
            ++qs.i_ffn_down;
        }
    } */ else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (name.find("attn_v.weight") != std::string::npos) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && name.find("attn_k.weight") != std::string::npos) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (name.find("ffn_down") != std::string::npos) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (name.find("attn_output.weight") != std::string::npos) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (name.find("attn_v.weight") != std::string::npos) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        } else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                   use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) {
            new_type = GGML_TYPE_Q6_K;
        } else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4)
            new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (name.find("attn_k.weight") != std::string::npos) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    
    } else if (name.find("attn_q.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (name.find("ffn_down") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
                }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (name.find("attn_output.weight") != std::string::npos) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
    
    } else if (name.find("attn_qkv.weight") != std::string::npos) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    } else if (name.find("ffn_gate") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    } else if (name.find("ffn_up") != std::string::npos) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

static size_t llama_tensor_quantize_impl(
    enum ggml_type new_type,
    const float * f32_data,
    const ggml_bf16_t * bf16_data,
    float tensor_scale,
    void * new_data,
    const int64_t chunk_size,
    int64_t nrows,
    int64_t n_per_row,
    const float * imatrix,
    std::vector<std::thread> & workers,
    const int nthread,
    const nvfp4_qa_tensor_ctx * qa_tensor,
    const char * tensor_name) {

    const bool do_qa = qa_tensor && qa_tensor->qa && qa_tensor->qa->file && new_type == GGML_TYPE_NVFP4;
    if (nrows <= 0 || n_per_row <= 0) return 0;

    const size_t row_size = ggml_row_size(new_type, n_per_row);
#ifdef GGML_USE_CUDA
    const bool nvfp4_cuda_direct = (new_type == GGML_TYPE_NVFP4) && (bf16_data || f32_data);
#endif
    auto nvfp4_progress_bar = [](double pct) {
        constexpr int k_bar_w = 28;
        const double p = std::max(0.0, std::min(100.0, pct));
        int fill = (int) std::llround((p / 100.0) * k_bar_w);
        fill = std::max(0, std::min(k_bar_w, fill));
        std::string bar((size_t) k_bar_w, '.');
        for (int i = 0; i < fill; ++i) bar[(size_t) i] = '=';
        if (fill > 0 && fill < k_bar_w) bar[(size_t) (fill - 1)] = '>';
        return bar;
    };
    auto nvfp4_format_eta = [](double sec) {
        char buf[64];
        if (!(sec >= 0.0) || !std::isfinite(sec)) {
            snprintf(buf, sizeof(buf), "n/a");
            return std::string(buf);
        }
        const int s = (int) std::llround(sec);
        if (s >= 3600) {
            snprintf(buf, sizeof(buf), "%dh %02dm %02ds", s / 3600, (s % 3600) / 60, s % 60);
        } else if (s >= 60) {
            snprintf(buf, sizeof(buf), "%dm %02ds", s / 60, s % 60);
        } else {
            snprintf(buf, sizeof(buf), "%ds", s);
        }
        return std::string(buf);
    };

    float nvfp4_a = NVFP4_A0;
    float nvfp4_b = NVFP4_B0;
    void * nvfp4_stream = reinterpret_cast<void *>(2); // per-thread default CUDA stream token
#ifdef GGML_USE_CUDA
    if (new_type == GGML_TYPE_NVFP4) {
        const int64_t nb_total  = (nrows * n_per_row) / QK_K;
        const int64_t sample_nb = llama_nvfp4_autotune_sample_blocks(nb_total);
        const int64_t sample_n  = sample_nb * QK_K;

        if (sample_n > 0) {
            const float inv_scale =
                (std::isfinite(tensor_scale) && tensor_scale > 0.0f) ? (1.0f / tensor_scale) : 1.0f;
            const int64_t nb_per_row = n_per_row / QK_K;
            const bool build_tune_qw = imatrix && nb_per_row > 0;
            const bool tune_unweighted_for_gate = tensor_name && strstr(tensor_name, "ffn_gate.weight") != nullptr;
            const bool use_tune_qw = build_tune_qw && !tune_unweighted_for_gate;

            std::vector<float> tune_x((size_t) sample_n);
            std::vector<float> tune_qw;
            const float * tune_qw_ptr = nullptr;
            if (use_tune_qw) {
                tune_qw.resize((size_t) sample_n);
                tune_qw_ptr = tune_qw.data();
            }

            int prep_threads = std::max(1, nthread);
            prep_threads = std::min<int>(prep_threads, (int) sample_nb);

            auto prep_worker = [&](int tid) {
                const int64_t ib0 = (sample_nb * tid) / prep_threads;
                const int64_t ib1 = (sample_nb * (tid + 1)) / prep_threads;
                for (int64_t ib = ib0; ib < ib1; ++ib) {
                    const int64_t src_block = llama_nvfp4_sample_block_index(ib, sample_nb, nb_total);

                    const int64_t src_off = src_block * QK_K;
                    const int64_t dst_off = ib * QK_K;

                    if (bf16_data) {
                        ggml_bf16_to_fp32_row(bf16_data + src_off, tune_x.data() + dst_off, QK_K);
                    } else {
                        memcpy(tune_x.data() + dst_off, f32_data + src_off, (size_t) QK_K * sizeof(float));
                    }

                    if (inv_scale != 1.0f) {
                        float * dst = tune_x.data() + dst_off;
                        for (int j = 0; j < QK_K; ++j) {
                            dst[j] *= inv_scale;
                        }
                    }

                    if (use_tune_qw) {
                        const int64_t ib_row = src_block % nb_per_row;
                        memcpy(tune_qw.data() + dst_off, imatrix + ib_row * QK_K, QK_K * sizeof(float));
                    }
                }
            };

            if (prep_threads > 1) {
                std::vector<std::thread> prep;
                prep.reserve((size_t) prep_threads - 1);
                for (int t = 1; t < prep_threads; ++t) {
                    prep.emplace_back(prep_worker, t);
                }
                prep_worker(0);
                for (auto & th : prep) {
                    th.join();
                }
            } else {
                prep_worker(0);
            }
            if (nvfp4_autotune_cuda(tune_x.data(), tune_qw_ptr, sample_n, &nvfp4_a, &nvfp4_b, nvfp4_stream)) {
                // tuned values are passed explicitly to CUDA quantize path
            }
        }
    }
#endif
    bool batched_cuda_success = false;
#ifdef GGML_USE_CUDA
    static constexpr size_t NVFP4_BATCHED_MAX_BYTES = (size_t) 1536 * 1024 * 1024;
    const size_t bytes_x_cuda = nvfp4_cuda_direct
        ? (size_t) nrows * (size_t) n_per_row * (bf16_data ? sizeof(ggml_bf16_t) : sizeof(float))
        : 0;
    const size_t bytes_y_cuda = nvfp4_cuda_direct ? (size_t) nrows * row_size : 0;
    const bool prefer_chunked_multistream = nvfp4_cuda_direct && (nthread > 1) && (nrows >= 4096);
    const bool can_fit_batched = nvfp4_cuda_direct && (bytes_x_cuda + bytes_y_cuda <= NVFP4_BATCHED_MAX_BYTES);
    const bool use_batched_cuda = can_fit_batched && !prefer_chunked_multistream;
    if (new_type == GGML_TYPE_NVFP4 && nrows > 0 && n_per_row > 0 && use_batched_cuda) {
        static thread_local struct {
            float learned_rate = 0.0f;  // rows/sec from previous tensors
            int tensor_count = 0;       // number of tensors processed
        } rate_tracker;

        auto t_start = std::chrono::high_resolution_clock::now();
        std::atomic<bool> batched_done{false};
        std::thread batched_progress;
        const bool show_batched_progress = nrows > 0;
        if (show_batched_progress) {
            const char * tn = tensor_name ? tensor_name : "(unknown)";
            const bool have_eta = rate_tracker.learned_rate > 0.0f;
            const double est_total = have_eta ? (double) nrows / rate_tracker.learned_rate : 0.0;
            LLAMA_LOG_INFO("nvfp4 quantize: start %s chunks=1 rows=%lld cols=%lld [batched]\n",
                tn, (long long) nrows, (long long) n_per_row);
            batched_progress = std::thread([&]() {
                static const char spin[] = {'|', '/', '-', '\\'};
                int si = 0;
                while (!batched_done.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(std::chrono::seconds(3));
                    if (batched_done.load(std::memory_order_relaxed)) {
                        break;
                    }
                    const auto now = std::chrono::high_resolution_clock::now();
                    const double elapsed = std::chrono::duration<double>(now - t_start).count();
                    const double pct = (have_eta && est_total > 0.0)
                        ? std::max(0.0, std::min(99.0, 100.0 * elapsed / est_total))
                        : 0.0;
                    const std::string bar = nvfp4_progress_bar(pct);
                    const std::string eta = (have_eta && est_total > 0.0)
                        ? nvfp4_format_eta(std::max(0.0, est_total - elapsed))
                        : "n/a";
                    LLAMA_LOG_CONT("\r\033[Knvfp4 quantize: %s %c [%s] %5.1f%% 0/1 elapsed %.1fs eta %s [batched]",
                        tn, spin[si], bar.c_str(), pct, elapsed, eta.c_str());
                    si = (si + 1) & 3;
                }
            });
        }

        // a_val / b_val removed; no-op

        if (bf16_data) {          // BF16 direct path
        const float x_scale = tensor_scale;
        if (nvfp4_quantize_cuda_ab(bf16_data, true, new_data, nrows, n_per_row,
                        imatrix, x_scale, nvfp4_a, nvfp4_b, nvfp4_stream)) 
                batched_cuda_success = true;
            
        }
    else if  (f32_data) {
    const float x_scale_f32 = tensor_scale;
    if (nvfp4_quantize_cuda_ab(f32_data, false, new_data, nrows, n_per_row,
                                imatrix, x_scale_f32, nvfp4_a, nvfp4_b, nvfp4_stream)) {
            batched_cuda_success = true;
    
        }
    }

        if (batched_cuda_success) {
            auto t_end = std::chrono::high_resolution_clock::now();
            float seconds = std::chrono::duration<float>(t_end - t_start).count();
            float rows_per_sec = seconds > 0.001f ? (float) nrows / seconds : (float) nrows * 1000.0f;

            batched_done.store(true, std::memory_order_relaxed);
            if (batched_progress.joinable()) {
                batched_progress.join();
            }
            if (show_batched_progress) {
                const std::string bar = nvfp4_progress_bar(100.0);
                LLAMA_LOG_CONT("\r\033[Knvfp4 quantize: %s [%s] 100.0%% 1/1 elapsed %.3fs eta 0s [batched]\n",
                    tensor_name ? tensor_name : "(unknown)", bar.c_str(), seconds);
            }

            if (rows_per_sec > 0.0f) {
                if (rate_tracker.learned_rate <= 0.0f) {
                    rate_tracker.learned_rate = rows_per_sec;
                } else {
                    rate_tracker.learned_rate = 0.3f * rows_per_sec + 0.7f * rate_tracker.learned_rate;
                }
                rate_tracker.tensor_count++;
            }

            return nrows * row_size;
        } else {
            batched_done.store(true, std::memory_order_relaxed);
            if (batched_progress.joinable()) {
                batched_progress.join();
            }
            LLAMA_LOG_INFO("nvfp4 batched: %s failed, falling back to chunked processing\n",
                tensor_name ? tensor_name : "(unknown)");
        }
    }
#endif
    const double log_every_s = 3.0;
    const int64_t nrows_per_chunk = chunk_size / n_per_row;
    const int64_t total_chunks = (nrows + nrows_per_chunk - 1) / nrows_per_chunk;

    const bool show_progress = (new_type == GGML_TYPE_NVFP4) && (total_chunks > 0);
    const auto progress_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start_steady = progress_start;
    std::atomic<int64_t> done_chunks{0};
    std::mutex progress_mutex;
    std::chrono::steady_clock::time_point last_log = progress_start;
    double last_pct_log = 0.0;
    bool logged_pass_eta = false;

    auto log_progress = [&](int64_t done, bool force) {
        if (!show_progress) return;
        auto now = std::chrono::steady_clock::now();

        const double elapsed = std::chrono::duration<double>(now - progress_start).count();
        const double since_last = std::chrono::duration<double>(now - last_log).count();
        
        const double pct = 100.0 * (double) done / (double) total_chunks;
        
        if (!force && since_last < log_every_s && (pct - last_pct_log < 5.0)) return;
        
        last_log = now;
        last_pct_log = pct;

        if (!logged_pass_eta && done > 0) {
            const double t_pass = elapsed / (double) done;
            LLAMA_LOG_INFO("nvfp4 quantize: %.2f seconds per pass - ETA ", t_pass);
            int total_seconds = (int) (t_pass * (double) total_chunks);
            if (total_seconds >= 60*60) {
                LLAMA_LOG_CONT("%d hours ", total_seconds / (60*60));
                total_seconds = total_seconds % (60*60);
            }
            LLAMA_LOG_CONT("%.2f minutes\n", total_seconds / 60.0);
            logged_pass_eta = true;
        }

        // Use steady state rate if possible (ignoring first chunk overhead)
        double rate;
        if (done > 1) {
             const double elapsed_steady = std::chrono::duration<double>(now - start_steady).count();
             rate = (double)(done - 1) / std::max(elapsed_steady, 1e-6);
        } else {
             rate = done > 0 ? (double) done / std::max(elapsed, 1e-6) : 0.0;
        }

        const double eta = rate > 0 ? (double) (total_chunks - done) / rate : 0.0;
        const std::string bar = nvfp4_progress_bar(pct);
        const std::string eta_s = (done > 1 && rate > 0)
            ? nvfp4_format_eta(eta)
            : std::string("n/a");
        LLAMA_LOG_CONT("\r\033[Knvfp4 quantize: %s [%s] %5.1f%% %lld/%lld elapsed %.1fs eta %s",
            tensor_name ? tensor_name : "(unknown)",
            bar.c_str(),
            pct,
            (long long) done, (long long) total_chunks,
            elapsed,
            eta_s.c_str());
    };

    if (show_progress) {
        LLAMA_LOG_INFO("nvfp4 quantize: start %s chunks=%lld rows=%lld cols=%lld\n",
            tensor_name ? tensor_name : "(unknown)",
            (long long) total_chunks, (long long) nrows, (long long) n_per_row);
        log_progress(0, true);
    }

    auto quantize_chunk = [&](int64_t first_row, int64_t this_nrow, size_t & this_size, std::vector<float> & fallback_buf) {
        this_size = 0;

        void * q_chunk = (char *) new_data + first_row * row_size;
        
        // NVFP4 BF16-direct CUDA fast path (chunked fallback)
        if (new_type == GGML_TYPE_NVFP4 && bf16_data) {
            const ggml_bf16_t * x_chunk = bf16_data + first_row * n_per_row;
#ifdef GGML_USE_CUDA
            if (!nvfp4_quantize_cuda_ab(x_chunk, true, q_chunk, this_nrow, n_per_row, imatrix, tensor_scale, nvfp4_a, nvfp4_b, nvfp4_stream)) {
                const size_t chunk_elems = (size_t) this_nrow * (size_t) n_per_row;
                if (fallback_buf.size() < chunk_elems) fallback_buf.resize(chunk_elems);
                ggml_bf16_to_fp32_row(x_chunk, fallback_buf.data(), (int64_t) chunk_elems);
                if (tensor_scale != 1.0f) {
                    const float inv_scale = 1.0f / tensor_scale;
                    for (size_t i = 0; i < chunk_elems; ++i) {
                        fallback_buf[i] *= inv_scale;
                    }
                }
                this_size = ggml_quantize_chunk(new_type, fallback_buf.data(), q_chunk, 0, this_nrow, n_per_row, imatrix);
            } else {
                this_size = (size_t) this_nrow * row_size;
            }
#else
            const size_t chunk_elems = (size_t) this_nrow * (size_t) n_per_row;
            if (fallback_buf.size() < chunk_elems) fallback_buf.resize(chunk_elems);
            ggml_bf16_to_fp32_row(x_chunk, fallback_buf.data(), (int64_t) chunk_elems);
            if (tensor_scale != 1.0f) {
                const float inv_scale = 1.0f / tensor_scale;
                for (size_t i = 0; i < chunk_elems; ++i) {
                    fallback_buf[i] *= inv_scale;
                }
            }
            this_size = ggml_quantize_chunk(new_type, fallback_buf.data(), q_chunk, 0, this_nrow, n_per_row, imatrix);
#endif
            return;
        }

        // CPU quant path (all other types, or NVFP4 CUDA fallback)
        if (f32_data) {
            if (new_type == GGML_TYPE_NVFP4 && tensor_scale != 1.0f) {
                const size_t chunk_elems = (size_t) this_nrow * (size_t) n_per_row;
                if (fallback_buf.size() < chunk_elems) fallback_buf.resize(chunk_elems);

                const float * x_chunk = f32_data + (size_t) first_row * (size_t) n_per_row;
                memcpy(fallback_buf.data(), x_chunk, chunk_elems * sizeof(float));

                const float inv_scale = 1.0f / tensor_scale;
                for (size_t i = 0; i < chunk_elems; ++i) {
                    fallback_buf[i] *= inv_scale;
                }

                this_size = ggml_quantize_chunk(new_type, fallback_buf.data(), q_chunk, 0, this_nrow, n_per_row, imatrix);
            } else {
                this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            }
            return;
        }

        if (bf16_data) { // still conver tto fp32 on cpu
            const ggml_bf16_t * x_chunk = bf16_data + first_row * n_per_row;
            const size_t chunk_elems = (size_t) this_nrow * (size_t) n_per_row;
            if (fallback_buf.size() < chunk_elems) fallback_buf.resize(chunk_elems);
            ggml_bf16_to_fp32_row(x_chunk, fallback_buf.data(), (int64_t) chunk_elems);
            if (new_type == GGML_TYPE_NVFP4 && tensor_scale != 1.0f) {
                const float inv_scale = 1.0f / tensor_scale;
                for (size_t i = 0; i < chunk_elems; ++i) {
                    fallback_buf[i] *= inv_scale;
                }
            }

            this_size = ggml_quantize_chunk(new_type, fallback_buf.data(), q_chunk, 0, this_nrow, n_per_row, imatrix);
            return;
        }

        // no source data at all -> hard fail
        this_size = 0;
    };

    // single-thread
    if (nthread < 2) {
        size_t new_size = 0;
        std::vector<float> fallback_buf;

        for (int64_t first_row = 0; first_row < nrows; first_row += nrows_per_chunk) {
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);

            size_t this_size = 0;
            quantize_chunk(first_row, this_nrow, this_size, fallback_buf);

            // if we ever got 0 here, that's a fatal condition for GGUF sizing
            if (this_size == 0) {
                throw std::runtime_error("quantize returned 0 bytes (missing source data or invalid path)");
            }

            new_size += this_size;

            if (do_qa && f32_data) {
                nvfp4_qa_stats st;
                const float * x_chunk = f32_data + first_row * n_per_row;
                const uint8_t * q_chunk = (const uint8_t *) new_data + (size_t) first_row * row_size;
                nvfp4_qa_accum(st, x_chunk, this_nrow, n_per_row, q_chunk);
                const int64_t chunk_id = qa_tensor->chunk_id ? qa_tensor->chunk_id->fetch_add(1) : first_row / nrows_per_chunk;
                nvfp4_qa_write_line(*qa_tensor->qa, qa_tensor->tensor_name, chunk_id, st);
            }

            GGML_UNUSED((char *) new_data + first_row * row_size);
            if (!ggml_validate_row_data(new_type, new_data, new_size)) {
                throw std::runtime_error("quantized data validation failed");
            }

            const int64_t done = done_chunks.fetch_add(1) + 1;
            if (show_progress) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                log_progress(done, false);
            }
        }

        if (show_progress) {
            std::lock_guard<std::mutex> lock(progress_mutex);
            log_progress(total_chunks, true);
            LLAMA_LOG_CONT("\n");
        }

        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        std::vector<float> fallback_buf;
        while (true) {
            int64_t first_row = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                first_row = counter;
                counter += nrows_per_chunk;
            }
            if (first_row >= nrows) {
                if (local_size > 0) {
                    std::unique_lock<std::mutex> lock(mutex);
                    new_size += local_size;
                }
                break;
            }

            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = 0;
            quantize_chunk(first_row, this_nrow, this_size, fallback_buf);
            if (this_size == 0) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
            local_size += this_size;

            // validate the quantized data
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }

            if (do_qa && f32_data) {
                nvfp4_qa_stats st;
                const float * x_chunk = f32_data + first_row * n_per_row;
                const uint8_t * q_chunk = (const uint8_t *) new_data + (size_t) first_row * row_size;
                nvfp4_qa_accum(st, x_chunk, this_nrow, n_per_row, q_chunk);
                const int64_t chunk_id = qa_tensor && qa_tensor->chunk_id ? qa_tensor->chunk_id->fetch_add(1) : first_row / nrows_per_chunk;
                nvfp4_qa_write_line(*qa_tensor->qa, qa_tensor->tensor_name, chunk_id, st);
            }

            const int64_t done = done_chunks.fetch_add(1) + 1;
            if (show_progress) {
                std::lock_guard<std::mutex> lock_progress(progress_mutex);
                log_progress(done, false);
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }

    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }

    if (show_progress) {
        std::lock_guard<std::mutex> lock(progress_mutex);
        log_progress(total_chunks, true);
        LLAMA_LOG_CONT("\n");
    }

    return new_size;
}

static bool tensor_type_requires_imatrix(const ggml_tensor * t, const ggml_type dst_type, const llama_ftype ftype) {
    return (
        dst_type == GGML_TYPE_IQ2_XXS || dst_type == GGML_TYPE_IQ2_XS ||
        dst_type == GGML_TYPE_IQ3_XXS || dst_type == GGML_TYPE_IQ1_S  ||
        dst_type == GGML_TYPE_IQ2_S   || dst_type == GGML_TYPE_IQ1_M  ||
        (   // Q2_K_S is the worst k-quant type - only allow it without imatrix for token embeddings
            dst_type == GGML_TYPE_Q2_K && ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && strcmp(t->name, "token_embd.weight") != 0
        )
    );
}

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    ggml_type default_type;
    llama_ftype ftype = params->ftype;

    switch (params->ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: default_type = GGML_TYPE_Q4_0; break;
        case LLAMA_FTYPE_MOSTLY_Q4_1: default_type = GGML_TYPE_Q4_1; break;
        case LLAMA_FTYPE_MOSTLY_Q5_0: default_type = GGML_TYPE_Q5_0; break;
        case LLAMA_FTYPE_MOSTLY_Q5_1: default_type = GGML_TYPE_Q5_1; break;
        case LLAMA_FTYPE_MOSTLY_Q8_0: default_type = GGML_TYPE_Q8_0; break;
        case LLAMA_FTYPE_MOSTLY_F16:  default_type = GGML_TYPE_F16;  break;
        case LLAMA_FTYPE_MOSTLY_BF16: default_type = GGML_TYPE_BF16; break;
        case LLAMA_FTYPE_ALL_F32:     default_type = GGML_TYPE_F32;  break;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: default_type = GGML_TYPE_MXFP4; break;
        case LLAMA_FTYPE_MOSTLY_NVFP4:     default_type = GGML_TYPE_NVFP4; break;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    default_type = GGML_TYPE_Q2_K;    break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  default_type = GGML_TYPE_Q3_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  default_type = GGML_TYPE_Q4_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  default_type = GGML_TYPE_Q5_K;    break;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    default_type = GGML_TYPE_Q6_K;    break;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   default_type = GGML_TYPE_TQ1_0;   break;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   default_type = GGML_TYPE_TQ2_0;   break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: default_type = GGML_TYPE_IQ2_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   default_type = GGML_TYPE_IQ2_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   default_type = GGML_TYPE_IQ2_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: default_type = GGML_TYPE_IQ3_XXS; break;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   default_type = GGML_TYPE_IQ1_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   default_type = GGML_TYPE_IQ1_M;   break;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  default_type = GGML_TYPE_IQ4_NL;  break;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  default_type = GGML_TYPE_IQ4_XS;  break;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:   default_type = GGML_TYPE_IQ3_S;   break;
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   default_type = GGML_TYPE_IQ3_S;   break;

        default: throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    llama_model_kv_override * kv_overrides = nullptr;
    if (params->kv_overrides) {
        auto * v = (std::vector<llama_model_kv_override>*)params->kv_overrides;
        kv_overrides = v->data();
    }

    std::vector<std::string> splits = {};
    llama_model_loader ml(fname_inp, splits, use_mmap, /*use_direct_io*/ true, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    llama_model model(llama_model_default_params());

    model.load_arch   (ml);
    model.load_hparams(ml);
    model.load_stats  (ml);

    quantize_state_impl qs(model, params);

    nvfp4_qa_context qa_ctx;

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        imatrix_data = static_cast<const std::unordered_map<std::string, std::vector<float>>*>(params->imatrix);
        if (imatrix_data) {
            LLAMA_LOG_INFO("================================ Have weights data with %d entries\n",int(imatrix_data->size()));
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        prune_list = *static_cast<const std::vector<int> *>(params->prune_layers);
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.meta.get());
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        const std::vector<llama_model_kv_override> & overrides = *(const std::vector<llama_model_kv_override> *)params->kv_overrides;
        for (const auto & o : overrides) {
            if (o.key[0] == 0) break;
            if (o.tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o.key, o.val_f64);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o.key, (uint32_t)std::abs(o.val_i64));
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o.key, o.val_bool);
            } else if (o.tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o.key, o.val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o.key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    struct moe_gate_up_merge_info {
        const llama_model_loader::llama_tensor_weight * gate = nullptr;
        std::string out_name;
    };
    std::unordered_map<const llama_model_loader::llama_tensor_weight *, moe_gate_up_merge_info> moe_gate_up_merges;

    const bool merge_moe_gate_up_for_nvfp4 =
        ftype == LLAMA_FTYPE_MOSTLY_NVFP4 &&
        params->tensor_types == nullptr &&
        params->token_embedding_type == GGML_TYPE_COUNT &&
        params->output_tensor_type == GGML_TYPE_COUNT;

    auto name_ends_with = [](const std::string & name, const std::string & suffix) -> bool {
        return name.size() >= suffix.size() &&
               name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    auto is_float_source_type = [](ggml_type t) -> bool {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
    };
    auto build_merged_gate_up_meta = [](const ggml_tensor * base, const std::string & out_name) -> ggml_tensor {
        ggml_tensor out = *base;
        ggml_set_name(&out, out_name.c_str());
        out.ne[1] *= 2;

        const size_t  type_size = ggml_type_size(out.type);
        const int64_t blck_size = ggml_blck_size(out.type);
        out.nb[0] = type_size;
        if (out.type == GGML_TYPE_NVFP4) {
            out.nb[1] = ggml_row_size(out.type, out.ne[0]);
        } else {
            out.nb[1] = out.nb[0]*(out.ne[0]/blck_size);
        }
        for (int i = 2; i < GGML_MAX_DIMS; ++i) {
            out.nb[i] = out.nb[i - 1]*out.ne[i - 1];
        }

        return out;
    };

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    if (merge_moe_gate_up_for_nvfp4) {
        std::unordered_map<std::string, const llama_model_loader::llama_tensor_weight *> by_name;
        by_name.reserve(tensors.size());
        for (const auto * it : tensors) {
            by_name.emplace(ggml_get_name(it->tensor), it);
        }

        std::unordered_set<const llama_model_loader::llama_tensor_weight *> skip;
        skip.reserve(tensors.size() / 16 + 1);

        const std::string suffix_up = "ffn_up_exps.weight";
        const std::string suffix_gate = "ffn_gate_exps.weight";
        const std::string suffix_gate_up = "ffn_gate_up_exps.weight";

        for (const auto * it : tensors) {
            const std::string up_name = ggml_get_name(it->tensor);
            if (!name_ends_with(up_name, suffix_up)) {
                continue;
            }

            const size_t suffix_pos = up_name.size() - suffix_up.size();
            std::string gate_name = up_name;
            gate_name.replace(suffix_pos, suffix_up.size(), suffix_gate);
            std::string merged_name = up_name;
            merged_name.replace(suffix_pos, suffix_up.size(), suffix_gate_up);

            auto gate_it = by_name.find(gate_name);
            if (gate_it == by_name.end()) {
                continue;
            }
            if (by_name.find(merged_name) != by_name.end()) {
                continue;
            }
            if (gate_it->second->idx != it->idx) {
                continue;
            }

            const ggml_tensor * up_t = it->tensor;
            const ggml_tensor * gate_t = gate_it->second->tensor;
            if (!is_float_source_type(up_t->type) || up_t->type != gate_t->type) {
                continue;
            }

            if (ggml_n_dims(up_t) < 3 || ggml_n_dims(gate_t) < 3) {
                continue;
            }
            if (up_t->ne[0] != gate_t->ne[0] ||
                up_t->ne[1] != gate_t->ne[1] ||
                up_t->ne[2] != gate_t->ne[2] ||
                up_t->ne[3] != gate_t->ne[3]) {
                continue;
            }

            moe_gate_up_merges[it] = { gate_it->second, merged_name };
            skip.insert(gate_it->second);

            LLAMA_LOG_INFO("%s: merging split MoE tensors %s + %s -> %s\n",
                __func__, gate_name.c_str(), up_name.c_str(), merged_name.c_str());
        }

        if (!skip.empty()) {
            std::vector<const llama_model_loader::llama_tensor_weight *> merged_tensors;
            merged_tensors.reserve(tensors.size() - skip.size());
            for (const auto * it : tensors) {
                if (skip.find(it) == skip.end()) {
                    merged_tensors.push_back(it);
                }
            }
            tensors.swap(merged_tensors);
        }
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    for (const auto * it : tensors) {
        const struct ggml_tensor * tensor = it->tensor;

        const std::string name = ggml_get_name(tensor);

        // TODO: avoid hardcoded tensor names - use the TN_* constants
        if (name.find("attn_v.weight")   != std::string::npos ||
            name.find("attn_qkv.weight") != std::string::npos ||
            name.find("attn_kv_b.weight")!= std::string::npos) {
            ++qs.n_attention_wv;
        } else if (name == LLM_TN(model.arch)(LLM_TENSOR_OUTPUT, "weight")) {
            qs.has_output = true;
        }
    }

    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)model.hparams.n_layer;

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int idx = 0;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> read_data_gate;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;
    std::vector<float> merged_f32_buf;
    std::vector<ggml_bf16_t> merged_bf16_buf;

    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // populate the original tensors so we get an initial meta data
    for (const auto * it : tensors) {
        uint16_t i_split = params->keep_split ? it->idx : 0;
        ggml_tensor * tensor = it->tensor;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }

        auto it_merge = moe_gate_up_merges.find(it);
        const bool is_merged_moe_gate_up = it_merge != moe_gate_up_merges.end();
        ggml_tensor merged_meta;
        if (is_merged_moe_gate_up) {
            merged_meta = build_merged_gate_up_meta(tensor, it_merge->second.out_name);
            gguf_add_tensor(ctx_outs[i_split].get(), &merged_meta);
        } else {
            gguf_add_tensor(ctx_outs[i_split].get(), tensor);
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    // Pre-register NVFP4 tensor-scale keys so metadata size is final before writing any data.
    {
        quantize_state_impl qs_meta(model, params);
        qs_meta.has_imatrix    = qs.has_imatrix;
        qs_meta.has_output     = qs.has_output;
        qs_meta.n_attention_wv = qs.n_attention_wv;
        qs_meta.n_ffn_down     = qs.n_ffn_down;
        qs_meta.n_ffn_gate     = qs.n_ffn_gate;
        qs_meta.n_ffn_up       = qs.n_ffn_up;

        for (const auto * it : tensors) {
            const struct ggml_tensor * tensor = it->tensor;
            auto it_merge = moe_gate_up_merges.find(it);
            const bool is_merged_moe_gate_up = it_merge != moe_gate_up_merges.end();
            ggml_tensor merged_meta;
            if (is_merged_moe_gate_up) {
                merged_meta = build_merged_gate_up_meta(tensor, it_merge->second.out_name);
                tensor = &merged_meta;
            }
            const std::string name = ggml_get_name(tensor);
            const uint16_t i_split = params->keep_split ? it->idx : 0;

            bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?
            quantize &= (ggml_n_dims(tensor) >= 2);
            quantize &= name.find("_norm.weight") == std::string::npos;
            quantize &= params->quantize_output_tensor || name != "output.weight";
            quantize &= !params->only_copy;
            quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;
            quantize &= name.find("altup")  == std::string::npos;
            quantize &= name.find("laurel") == std::string::npos;
            quantize &= name.find("per_layer_model_proj") == std::string::npos;
            quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
            quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");
            quantize &= name.find("ssm_conv1d.weight") == std::string::npos;
            quantize &= name.find("shortconv.conv.weight") == std::string::npos;
            quantize &= name.find("time_mix_first.weight") == std::string::npos;
            quantize &= name.find("time_mix_w0.weight") == std::string::npos;
            quantize &= name.find("time_mix_w1.weight") == std::string::npos;
            quantize &= name.find("time_mix_w2.weight") == std::string::npos;
            quantize &= name.find("time_mix_v0.weight") == std::string::npos;
            quantize &= name.find("time_mix_v1.weight") == std::string::npos;
            quantize &= name.find("time_mix_v2.weight") == std::string::npos;
            quantize &= name.find("time_mix_a0.weight") == std::string::npos;
            quantize &= name.find("time_mix_a1.weight") == std::string::npos;
            quantize &= name.find("time_mix_a2.weight") == std::string::npos;
            quantize &= name.find("time_mix_g1.weight") == std::string::npos;
            quantize &= name.find("time_mix_g2.weight") == std::string::npos;
            quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
            quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
            quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;
            quantize &= name.find("attn_rel_b.weight") == std::string::npos;
            quantize &= name.find(".position_embd.") == std::string::npos;

            ggml_type new_type = default_type;
            if (quantize) {
                if (!params->pure && ggml_is_quantized(default_type)) {
                    int fallback = qs_meta.n_fallback;
                    new_type = llama_tensor_get_type(qs_meta, new_type, tensor, ftype);
                    if (params->tensor_types && qs_meta.n_fallback - fallback == 0) {
                        const std::vector<tensor_quantization> & tensor_types = *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
                        const std::string tensor_name(tensor->name);
                        for (const auto & [tname, qtype] : tensor_types) {
                            if (std::regex pattern(tname); std::regex_search(tensor_name, pattern)) {
                                new_type = qtype;
                            }
                        }
                    }
                }
                if (params->token_embedding_type < GGML_TYPE_COUNT && strcmp(tensor->name, "token_embd.weight") == 0) {
                    new_type = params->token_embedding_type;
                }
                if (params->output_tensor_type < GGML_TYPE_COUNT && strcmp(tensor->name, "output.weight") == 0) {
                    new_type = params->output_tensor_type;
                }

                quantize = tensor->type != new_type;

                if (quantize && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) && ggml_is_quantized(new_type)) {
                    const size_t src_nbytes = ggml_nbytes(tensor);
                    const size_t dst_nbytes = (size_t) ggml_row_size(new_type, tensor->ne[0]) * (size_t) tensor->ne[1] * (size_t) tensor->ne[2] * (size_t) tensor->ne[3];
                    if (src_nbytes > dst_nbytes && (src_nbytes - dst_nbytes) < LLAMA_QUANT_MIN_SAVINGS_BYTES) {
                        quantize = false;
                        new_type = tensor->type;
                    }
                }
            }

            if (is_merged_moe_gate_up) {
                new_type = GGML_TYPE_NVFP4;
                quantize = true;
            }

            if (!quantize) {
                new_type = tensor->type;
            }

            if (new_type == GGML_TYPE_NVFP4) {
                const std::string k1 = name + ".tensor_scale";
                gguf_set_val_f32(ctx_outs[i_split].get(), k1.c_str(), 1.0f);
            }
        }
    }

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    const auto tn = LLM_TN(model.arch);

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    // flag for `--dry-run`, to let the user know if imatrix will be required for a real
    // quantization, as a courtesy
    bool will_require_imatrix = false;

    for (const auto * it : tensors) {
        const auto & weight = *it;
        ggml_tensor * tensor = weight.tensor;
        auto it_merge = moe_gate_up_merges.find(it);
        const bool is_merged_moe_gate_up = it_merge != moe_gate_up_merges.end();
        ggml_tensor merged_meta;
        const ggml_tensor * tensor_meta = tensor;
        if (is_merged_moe_gate_up) {
            merged_meta = build_merged_gate_up_meta(tensor, it_merge->second.out_name);
            tensor_meta = &merged_meta;
        }
        ggml_tensor * gate_tensor = is_merged_moe_gate_up ? it_merge->second.gate->tensor : nullptr;

        if (weight.idx != cur_split && params->keep_split) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const std::string name = ggml_get_name(tensor_meta);

        const bool tensor_needs_staging = !ml.use_mmap ||
            (ml.use_mmap && ml.has_nvfp4_tensor_scales && tensor->type == GGML_TYPE_NVFP4);
        if (tensor_needs_staging) {
            if (read_data.size() < ggml_nbytes(tensor)) {
                read_data.resize(ggml_nbytes(tensor));
            }
            ml.load_data_for(tensor);
        }
        ml.load_data_for(tensor);
        if (is_merged_moe_gate_up) {
            GGML_ASSERT(gate_tensor != nullptr);
            const bool gate_needs_staging = !ml.use_mmap ||
                (ml.use_mmap && ml.has_nvfp4_tensor_scales && gate_tensor->type == GGML_TYPE_NVFP4);
            if (gate_needs_staging) {
                if (read_data_gate.size() < ggml_nbytes(gate_tensor)) {
                    read_data_gate.resize(ggml_nbytes(gate_tensor));
                }
                gate_tensor->data = read_data_gate.data();
            }
            ml.load_data_for(gate_tensor);
        }

        LLAMA_LOG_INFO("[%4d/%4d] %36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               name.c_str(),
               llama_format_tensor_shape(tensor_meta).c_str(),
               ggml_type_name(tensor_meta->type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

        // quantize only 2D and 3D tensors (experts)
        quantize &= (ggml_n_dims(tensor_meta) >= 2);

        // do not quantize norm tensors
        quantize &= name.find("_norm.weight") == std::string::npos;

        quantize &= params->quantize_output_tensor || name != "output.weight";
        quantize &= !params->only_copy;

        // do not quantize expert gating tensors
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

        // these are very small (e.g. 4x4)
        quantize &= name.find("altup")  == std::string::npos;
        quantize &= name.find("laurel") == std::string::npos;

        // these are not too big so keep them as it is
        quantize &= name.find("per_layer_model_proj") == std::string::npos;

        // do not quantize positional embeddings and token types (BERT)
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_POS_EMBD,    "weight");
        quantize &= name != LLM_TN(model.arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

        // do not quantize Mamba /Kimi's small conv1d weights
        // NOTE: can't use LLM_TN here because the layer number is not known
        quantize &= name.find("ssm_conv1d") == std::string::npos;
        quantize &= name.find("shortconv.conv.weight") == std::string::npos;

        // do not quantize RWKV's small yet 2D weights
        quantize &= name.find("time_mix_first.weight") == std::string::npos;
        quantize &= name.find("time_mix_w0.weight") == std::string::npos;
        quantize &= name.find("time_mix_w1.weight") == std::string::npos;
        quantize &= name.find("time_mix_w2.weight") == std::string::npos;
        quantize &= name.find("time_mix_v0.weight") == std::string::npos;
        quantize &= name.find("time_mix_v1.weight") == std::string::npos;
        quantize &= name.find("time_mix_v2.weight") == std::string::npos;
        quantize &= name.find("time_mix_a0.weight") == std::string::npos;
        quantize &= name.find("time_mix_a1.weight") == std::string::npos;
        quantize &= name.find("time_mix_a2.weight") == std::string::npos;
        quantize &= name.find("time_mix_g1.weight") == std::string::npos;
        quantize &= name.find("time_mix_g2.weight") == std::string::npos;
        quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
        quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
        quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

        // do not quantize relative position bias (T5)
        quantize &= name.find("attn_rel_b.weight") == std::string::npos;

        // do not quantize specific multimodal tensors
        quantize &= name.find(".position_embd.") == std::string::npos;

        ggml_type new_type;
        void * new_data;
        size_t new_size;
        bool nvfp4_quantized = false;
        float tensor_scale = 1.0f;

        if (quantize) {
            new_type = default_type;

            // get more optimal quantization type based on the tensor shape, layer, etc.
            if (!params->pure && ggml_is_quantized(default_type)) {
                // if the user provided tensor types - use those
                bool manual = false;
                if (params->tensor_types) {
                    const std::vector<tensor_quantization> & tensor_types = *static_cast<const std::vector<tensor_quantization> *>(params->tensor_types);
                    const std::string tensor_name(tensor->name);
                    for (const auto & [tname, qtype] : tensor_types) {
                        if (std::regex pattern(tname); std::regex_search(tensor_name, pattern)) {
                            if  (qtype != new_type) {
                                LLAMA_LOG_WARN("(manual override: %s -> %s) ", ggml_type_name(new_type), ggml_type_name(qtype));
                                new_type = qtype; // if two or more types are specified for the same tensor, the last match wins
                                manual = true;
                                break;
                            }
                        }
                    }
                }

                // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
                    if (!manual) {
                        new_type = llama_tensor_get_type(qs, new_type, tensor_meta, ftype);
                    }

                // incompatible tensor shapes are handled here - fallback to a compatible type
                {
                    bool convert_incompatible_tensor = false;

                    const int64_t nx = tensor_meta->ne[0];
                    const int64_t ny = tensor_meta->ne[1];
                    const int64_t qk_k = ggml_blck_size(new_type);

                    if (nx % qk_k != 0) {
                        LLAMA_LOG_WARN("\n\n%s : tensor cols %" PRId64 " x %" PRId64 " are not divisible by %" PRId64 ", required for %s", __func__, nx, ny, qk_k, ggml_type_name(new_type));
                        convert_incompatible_tensor = true;
                    } else {
                        ++qs.n_k_quantized;
                    }

                    if (convert_incompatible_tensor) {
                        switch (new_type) {
                            case GGML_TYPE_TQ1_0:
                            case GGML_TYPE_TQ2_0:  new_type = GGML_TYPE_Q4_0; break;  // TODO: use a symmetric type instead
                            case GGML_TYPE_IQ2_XXS:
                            case GGML_TYPE_IQ2_XS:
                            case GGML_TYPE_IQ2_S:
                            case GGML_TYPE_IQ3_XXS:
                            case GGML_TYPE_IQ3_S:
                            case GGML_TYPE_IQ1_S:
                            case GGML_TYPE_IQ1_M:
                            case GGML_TYPE_Q2_K:
                            case GGML_TYPE_Q3_K:
                            case GGML_TYPE_IQ4_XS: new_type = GGML_TYPE_IQ4_NL; break;
                            case GGML_TYPE_Q4_K:   new_type = GGML_TYPE_Q5_0;   break;
                            case GGML_TYPE_Q5_K:   new_type = GGML_TYPE_Q5_1;   break;
                            case GGML_TYPE_Q6_K:   new_type = GGML_TYPE_Q8_0;   break;
                            case GGML_TYPE_NVFP4: new_type =
                            GGML_TYPE_Q4_K; break;

                            default: throw std::runtime_error("\nUnsupported tensor size encountered\n");
                        }
                        if (tensor_meta->ne[0] % ggml_blck_size(new_type) != 0) {
                            new_type = GGML_TYPE_F16;
                        }
                        LLAMA_LOG_WARN(" - using fallback quantization %s\n", ggml_type_name(new_type));
                        ++qs.n_fallback;
                    }
                }
            }
            if (params->token_embedding_type < GGML_TYPE_COUNT && name == "token_embd.weight") {
                new_type = params->token_embedding_type;
            }
            if (params->output_tensor_type < GGML_TYPE_COUNT && name == "output.weight") {
                new_type = params->output_tensor_type;
            }

            // no point to quantize if we only save 1mb.
            if (!is_merged_moe_gate_up &&
                tensor_meta->type != new_type &&
                (tensor_meta->type == GGML_TYPE_F32 || tensor_meta->type == GGML_TYPE_F16 || tensor_meta->type == GGML_TYPE_BF16) &&
                ggml_is_quantized(new_type)) {
                const size_t src_nbytes = ggml_nbytes(tensor_meta);
                const size_t dst_nbytes = (size_t) ggml_row_size(new_type, tensor_meta->ne[0]) * (size_t) tensor_meta->ne[1] * (size_t) tensor_meta->ne[2] * (size_t) tensor_meta->ne[3];
                if (src_nbytes > dst_nbytes && (src_nbytes - dst_nbytes) < LLAMA_QUANT_MIN_SAVINGS_BYTES) {
                    new_type = tensor_meta->type;
                }
            }

            // If we've decided to quantize to the same type the tensor is already
            // in then there's nothing to do.
            quantize = tensor_meta->type != new_type;
        }

        if (is_merged_moe_gate_up) {
            new_type = GGML_TYPE_NVFP4;
            quantize = true;
        }

        if (!quantize) {
            new_type = tensor_meta->type;
            new_data = tensor->data;
            new_size = ggml_nbytes(tensor_meta);
            LLAMA_LOG_INFO("size = %8.3f MiB\n", ggml_nbytes(tensor_meta)/1024.0/1024.0);
        } else {
            const int64_t nelements = ggml_nelements(tensor_meta);

            const float * imatrix = nullptr;
            if (imatrix_data) {
                const char * imatrix_name = is_merged_moe_gate_up ? tensor->name : tensor_meta->name;
                auto it = imatrix_data->find(remap_imatrix(imatrix_name, mapped));
                if (it == imatrix_data->end()) {
                    LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, name.c_str());
                } else {
                    const size_t expected_imatrix = (size_t) tensor_meta->ne[0] * (size_t) tensor_meta->ne[2] * (size_t) tensor_meta->ne[3];
                    if (it->second.size() == expected_imatrix) {
                        imatrix = it->second.data();
                    } else {
                        LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                int(it->second.size()), int(expected_imatrix), name.c_str());

                        // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                        // this is a significant error and it may be good idea to abort the process if this happens,
                        // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                        // tok_embd should be ignored in this case, since it always causes this warning
                        if (name != tn(LLM_TENSOR_TOKEN_EMBD, "weight")) {
                            throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                    int(it->second.size()), int(expected_imatrix), name.c_str()));
                        }
                    }
                }
            }
            if ((new_type == GGML_TYPE_IQ2_XXS ||
                 new_type == GGML_TYPE_IQ2_XS  ||
                 new_type == GGML_TYPE_IQ2_S   ||
                 new_type == GGML_TYPE_IQ1_S   ||
                (new_type == GGML_TYPE_IQ1_M && name != "token_embd.weight" && name != "output.weight")  ||
                (new_type == GGML_TYPE_Q2_K && params->ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && name != "token_embd.weight")) && !imatrix) {
                LLAMA_LOG_ERROR("\n\n============================================================\n");
                LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", name.c_str());
                LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                LLAMA_LOG_ERROR("============================================================\n\n");
                throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", name.c_str()));
            }

            float * f32_data = nullptr;
            const ggml_bf16_t * bf16_data = nullptr;
            if (is_merged_moe_gate_up) {
                const size_t src_slice_elems = (size_t) tensor->ne[0] * (size_t) tensor->ne[1];
                const size_t dst_slice_elems = src_slice_elems * 2;
                const size_t n_slices = (size_t) tensor->ne[2] * (size_t) tensor->ne[3];

                if (tensor->type == GGML_TYPE_BF16 && new_type == GGML_TYPE_NVFP4 && !qa_ctx.file) {
                    merged_bf16_buf.resize((size_t) nelements);
                    const auto * gate_src = (const ggml_bf16_t *) gate_tensor->data;
                    const auto * up_src   = (const ggml_bf16_t *) tensor->data;
                    auto * dst = merged_bf16_buf.data();
                    for (size_t is = 0; is < n_slices; ++is) {
                        std::memcpy(dst + is*dst_slice_elems, gate_src + is*src_slice_elems, src_slice_elems*sizeof(ggml_bf16_t));
                        std::memcpy(dst + is*dst_slice_elems + src_slice_elems, up_src + is*src_slice_elems, src_slice_elems*sizeof(ggml_bf16_t));
                    }
                    bf16_data = merged_bf16_buf.data();
                } else {
                    merged_f32_buf.resize((size_t) nelements);
                    auto convert_slice = [&](const ggml_tensor * src_t, size_t src_offs, size_t n, float * dst) {
                        if (src_t->type == GGML_TYPE_F32) {
                            const auto * src = (const float *) src_t->data + src_offs;
                            std::memcpy(dst, src, n*sizeof(float));
                        } else if (src_t->type == GGML_TYPE_F16) {
                            const auto * src = (const ggml_fp16_t *) src_t->data + src_offs;
                            for (size_t i = 0; i < n; ++i) {
                                dst[i] = ggml_fp16_to_fp32(src[i]);
                            }
                        } else if (src_t->type == GGML_TYPE_BF16) {
                            const auto * src = (const ggml_bf16_t *) src_t->data + src_offs;
                            for (size_t i = 0; i < n; ++i) {
                                dst[i] = ggml_bf16_to_fp32(src[i]);
                            }
                        } else {
                            GGML_ABORT("unsupported source type for merged MoE gate_up: %s", ggml_type_name(src_t->type));
                        }
                    };

                    for (size_t is = 0; is < n_slices; ++is) {
                        const size_t src_offs = is*src_slice_elems;
                        float * dst = merged_f32_buf.data() + is*dst_slice_elems;
                        convert_slice(gate_tensor, src_offs, src_slice_elems, dst);
                        convert_slice(tensor, src_offs, src_slice_elems, dst + src_slice_elems);
                    }
                    f32_data = merged_f32_buf.data();
                }
            } else if (tensor->type == GGML_TYPE_BF16 && new_type == GGML_TYPE_NVFP4 && !qa_ctx.file) {
                bf16_data = (const ggml_bf16_t *) tensor->data;
            }

            if (!bf16_data && !is_merged_moe_gate_up) {
                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }
            }

            const float * f32_quant_data = f32_data;
            if (new_type == GGML_TYPE_NVFP4) {
                nvfp4_quantized = true;
                float absmax = 0.0f;

                if (bf16_data) {
                    absmax = llama_tensor_absmax_bf16_mt(bf16_data, (size_t) nelements, nthread);
                } else {
                    absmax = llama_tensor_absmax_f32_mt(f32_data, (size_t) nelements, nthread);
                }

                const float cap_ts = LLAMA_NVFP4_TENSOR_CAP_FIXED;
                tensor_scale = absmax / (LLAMA_NVFP4_MAX_FP4 * cap_ts);
                if (!std::isfinite(tensor_scale) || tensor_scale <= 0.0f) {
                    tensor_scale = 1.0f;
                }
            }
            const int64_t n_per_row = tensor_meta->ne[0];
            const int64_t nrows = tensor_meta->ne[1];
            const size_t new_data_size =
                (size_t) ggml_row_size(new_type, n_per_row) *
                (size_t) nrows *
                (size_t) tensor_meta->ne[2] *
                (size_t) tensor_meta->ne[3];

            if (work.size() < new_data_size) {
                work.resize(new_data_size);
            }
            new_data = work.data();

            static const int64_t min_chunk_size = 32 * 512;
            int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

            const int64_t nelements_matrix = tensor_meta->ne[0] * tensor_meta->ne[1];
#ifdef GGML_USE_CUDA
            if (new_type == GGML_TYPE_NVFP4 && (bf16_data || f32_data)) {
                const int64_t target_rows = std::clamp<int64_t>(
                    nelements_matrix / std::max<int64_t>(1, (int64_t) nthread * 6),
                    256,
                    1024);
                const int64_t tuned_chunk_size = target_rows * n_per_row;
                chunk_size = std::max(chunk_size, tuned_chunk_size);
            }
#endif
            const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
            int64_t nthread_use = nthread > 1 ? std::max((int64_t) 1, std::min((int64_t) nthread, nchunk)) : 1;
#ifdef GGML_USE_CUDA
            if (new_type == GGML_TYPE_NVFP4 && (bf16_data || f32_data)) {
                nthread_use = std::min<int64_t>(nthread_use, 8);
            }
#endif

            new_size = 0;
            std::atomic<int64_t> qa_chunk_id{0};
            nvfp4_qa_tensor_ctx qa_tensor;
            if (qa_ctx.file && new_type == GGML_TYPE_NVFP4) {
                qa_tensor.qa = &qa_ctx;
                qa_tensor.tensor_name = tensor_meta->name;
                qa_tensor.chunk_id = &qa_chunk_id;
                qa_tensor.n_per_row = n_per_row;
            }

            const int64_t ne2 = tensor_meta->ne[2];
            const int64_t ne3 = tensor_meta->ne[3];
            const size_t slice_elems = (size_t) nelements_matrix;
            const size_t slice_bytes = (size_t) ggml_row_size(new_type, n_per_row) * (size_t) nrows;

            const int64_t n_slices = ne2 * ne3;
            const bool parallel_nvfp4_slices = (new_type == GGML_TYPE_NVFP4) && (n_slices > 1) && !qa_ctx.file;

            if (parallel_nvfp4_slices) {
                const int slice_threads = (int) std::min<int64_t>(std::min<int64_t>(8, nthread), n_slices);
                std::atomic<int64_t> next_slice{0};
                std::atomic<size_t> new_size_atomic{0};
                std::atomic<bool> failed{false};
                std::mutex err_mtx;
                std::string err_msg;

                auto run_slice = [&](int /*tid*/) {
                    std::vector<std::thread> local_workers;
                    while (true) {
                        if (failed.load(std::memory_order_acquire)) {
                            return;
                        }

                        const int64_t is = next_slice.fetch_add(1, std::memory_order_relaxed);
                        if (is >= n_slices) {
                            return;
                        }

                        const float * f32_data_s = f32_quant_data ? (f32_quant_data + (size_t) is * slice_elems) : nullptr;
                        const ggml_bf16_t * bf16_data_s = bf16_data ? (bf16_data + (size_t) is * slice_elems) : nullptr;
                        void * new_data_s = (char *) new_data + (size_t) is * slice_bytes;
                        const float * imatrix_s = imatrix ? (imatrix + (size_t) is * (size_t) n_per_row) : nullptr;

                        try {
                            const size_t slice_size = llama_tensor_quantize_impl(
                                new_type, f32_data_s, bf16_data_s, tensor_scale,
                                new_data_s, chunk_size, nrows, n_per_row, imatrix_s, local_workers, 1,
                                (qa_ctx.file && new_type == GGML_TYPE_NVFP4) ? &qa_tensor : nullptr, tensor_meta->name);
                            new_size_atomic.fetch_add(slice_size, std::memory_order_relaxed);
                        } catch (const std::exception & e) {
                            std::lock_guard<std::mutex> lock(err_mtx);
                            if (!failed.exchange(true, std::memory_order_acq_rel)) {
                                err_msg = e.what();
                            }
                            return;
                        }
                    }
                };

                std::vector<std::thread> slice_workers;
                slice_workers.reserve((size_t) std::max(0, slice_threads - 1));
                for (int t = 1; t < slice_threads; ++t) {
                    slice_workers.emplace_back(run_slice, t);
                }
                run_slice(0);
                for (auto & th : slice_workers) {
                    th.join();
                }

                if (failed.load(std::memory_order_acquire)) {
                    throw std::runtime_error(err_msg.empty() ? "parallel NVFP4 slice quantization failed" : err_msg);
                }

                new_size += new_size_atomic.load(std::memory_order_relaxed);
            } else {
                for (int64_t i04 = 0; i04 < ne3; ++i04) {
                    for (int64_t i03 = 0; i03 < ne2; ++i03) {
                        const int64_t is = i03 + ne2 * i04;

                        const float * f32_data_s = f32_quant_data ? (f32_quant_data + (size_t) is * slice_elems) : nullptr;
                        const ggml_bf16_t * bf16_data_s = bf16_data ? (bf16_data + (size_t) is * slice_elems) : nullptr;
                        void * new_data_s = (char *) new_data + (size_t) is * slice_bytes;
                        const float * imatrix_s = imatrix ? (imatrix + (size_t) is * (size_t) n_per_row) : nullptr;

                        new_size += llama_tensor_quantize_impl(
                            new_type, f32_data_s, bf16_data_s, tensor_scale,
                            new_data_s, chunk_size, nrows, n_per_row, imatrix_s, workers, nthread_use,
                            (qa_ctx.file && new_type == GGML_TYPE_NVFP4) ? &qa_tensor : nullptr, tensor_meta->name);
                    }
                }
            }

            const size_t src_nbytes = ggml_nbytes(tensor) + (is_merged_moe_gate_up ? ggml_nbytes(gate_tensor) : 0);
            LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", src_nbytes/1024.0/1024.0, new_size/1024.0/1024.0);
        }
        total_size_org += ggml_nbytes(tensor) + (is_merged_moe_gate_up ? ggml_nbytes(gate_tensor) : 0);
        total_size_new += new_size;

        // update the gguf meta data as we go
        gguf_set_tensor_type(ctx_outs[cur_split].get(), name.c_str(), new_type);
        if (nvfp4_quantized) {
            const std::string k1 = name + ".tensor_scale";

            gguf_set_val_f32(ctx_outs[cur_split].get(), k1.c_str(), tensor_scale);
        }

        if (new_type != GGML_TYPE_NVFP4) {
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), name.c_str())) == new_size);
        }
        gguf_set_tensor_data(ctx_outs[cur_split].get(), name.c_str(), new_data);

        if (!params->dry_run) {
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        }
    } // iterate over tensors

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, qs.n_k_quantized + qs.n_fallback);
    }

    if (qa_ctx.file) {
        std::fclose(qa_ctx.file);
        qa_ctx.file = nullptr;
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q5_1,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tensor_types                =*/ nullptr,
        /*.prune_layers                =*/ nullptr
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}
