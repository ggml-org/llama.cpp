#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "self-spec-bias-sampler.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(int /*argc*/, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f input.txt --stream-interval 3 --draft-bias-beta 0.2\n", argv[0]);
    LOG("\n");
}

static size_t spec_bias_lcp(const llama_tokens & a, const llama_tokens & b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) {
        i++;
    }
    return i;
}

// drop the kv cache that the new prompt cannot reuse, return how many tokens are kept
static size_t spec_bias_prompt_reuse(
        llama_context * ctx,
        llama_seq_id    seq_id,
        const llama_tokens & cache,
        const llama_tokens & prompt) {
    size_t n_keep = spec_bias_lcp(cache, prompt);

    // one token must stay unevaluated, otherwise there are no logits to sample from
    if (n_keep > 0 && n_keep == prompt.size()) {
        n_keep--;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx), seq_id, n_keep, -1)) {
        llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
        n_keep = 0;
    }

    return n_keep;
}

// split on whitespace, then rebuild the growing prefixes of the line
static std::vector<std::string> spec_bias_prefixes(const std::string & line, int interval) {
    std::vector<std::string> words;

    for (size_t i = 0; i < line.size(); ) {
        while (i < line.size() && std::isspace((unsigned char) line[i])) {
            i++;
        }
        const size_t beg = i;
        while (i < line.size() && !std::isspace((unsigned char) line[i])) {
            i++;
        }
        if (i > beg) {
            words.push_back(line.substr(beg, i - beg));
        }
    }

    std::vector<std::string> res;

    if (words.empty()) {
        return res;
    }

    if (interval <= 0) {
        res.push_back(line);
        return res;
    }

    std::string cur;
    for (size_t i = 0; i < words.size(); ++i) {
        cur += (i == 0 ? "" : " ") + words[i];

        if ((i + 1) % interval == 0) {
            res.push_back(cur);
        }
    }

    if (res.empty() || res.back() != cur) {
        res.push_back(cur);
    }

    return res;
}

struct spec_bias_record {
    std::string id;
    std::string source;

    nlohmann::ordered_json segmentation;

    std::vector<std::string> stream_ins;
};

// Build the work list.
//
// A .jsonl input carries the segmentation already, which is how a caller plugs
// in its own policy. Plain text is split here with --stream-interval instead.
static std::vector<spec_bias_record> spec_bias_load(const common_params & params) {
    std::vector<spec_bias_record> res;

    std::string stem = params.prompt_file;
    {
        const size_t slash = stem.find_last_of("/\\");
        if (slash != std::string::npos) {
            stem = stem.substr(slash + 1);
        }
        if (stem.empty()) {
            stem = "line";
        }
    }

    const bool is_jsonl = stem.size() > 6 && stem.compare(stem.size() - 6, 6, ".jsonl") == 0;

    // params.prompt has been through string_process_escapes, which would eat the
    // backslashes in json, so a stream file is read straight from disk
    std::vector<std::string> lines;

    if (is_jsonl) {
        std::ifstream f(params.prompt_file);
        if (!f) {
            LOG_ERR("%s: cannot open %s\n", __func__, params.prompt_file.c_str());
            return {};
        }
        for (std::string l; std::getline(f, l); ) {
            lines.push_back(l);
        }
    } else {
        lines = string_split<std::string>(params.prompt, '\n');
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string & line = lines[i];

        if (line.find_first_not_of(" \t\r") == std::string::npos) {
            continue;
        }

        spec_bias_record rec;

        if (is_jsonl) {
            nlohmann::ordered_json j;
            try {
                j = nlohmann::ordered_json::parse(line);
            } catch (const std::exception & e) {
                LOG_ERR("%s: %s line %zu is not json: %s\n", __func__, params.prompt_file.c_str(), i + 1, e.what());
                return {};
            }

            if (!j.contains("stream_ins") || !j["stream_ins"].is_array() || j["stream_ins"].empty()) {
                LOG_ERR("%s: %s line %zu has no stream_ins\n", __func__, params.prompt_file.c_str(), i + 1);
                return {};
            }

            rec.stream_ins   = j["stream_ins"].get<std::vector<std::string>>();
            rec.id           = j.value("id", stem + ":" + std::to_string(i));
            rec.source       = j.value("source", rec.stream_ins.back());
            rec.segmentation = j.value("segmentation", nlohmann::ordered_json::object());
        } else {
            rec.stream_ins = spec_bias_prefixes(line, params.spec_bias_stream_interval);
            if (rec.stream_ins.empty()) {
                continue;
            }
            rec.id           = stem + ":" + std::to_string(i);
            rec.source       = line;
            rec.segmentation = {
                { "policy", "interval" },
                { "n",      params.spec_bias_stream_interval },
            };
        }

        res.push_back(std::move(rec));
    }

    return res;
}

// Check a draft against the model with one decode.
//
// The batch is [id_last, draft...] and every position asks for logits, so the
// logits at index i hold the model's prediction for draft[i]. Sampling walks
// forward while the model agrees with the draft. On the first disagreement the
// remaining draft tokens are dropped from the kv cache.
//
// Returns the accepted draft tokens plus one more token to continue from.
static llama_tokens spec_bias_verify_draft(
        llama_context * ctx,
        llama_sampler * smpl,
        llama_batch   & batch,
        llama_seq_id    seq_id,
        llama_token     id_last,
        const llama_tokens & draft,
        int           & n_past) {
    GGML_ASSERT(!draft.empty());

    const int n_past_batch = n_past;

    // id_last plus the draft must fit, common_batch_add aborts on overflow
    const size_t n_draft = std::min(draft.size(), (size_t) llama_n_batch(ctx) - 1);

    common_batch_clear(batch);
    common_batch_add(batch, id_last, n_past, { seq_id }, true);

    for (size_t i = 0; i < n_draft; ++i) {
        common_batch_add(batch, draft[i], n_past + 1 + i, { seq_id }, true);
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: failed to decode the draft\n", __func__);
        return {};
    }

    spec_bias_sampler_set_seq(spec_bias_sampler_of(smpl), draft, 0);

    llama_tokens ids;
    ids.reserve(draft.size() + 1);

    // llama_sampler_sample also accepts the token, so the sampler advances on its own
    size_t i = 0;
    for (; i < n_draft; ++i) {
        const llama_token id = llama_sampler_sample(smpl, ctx, (int32_t) i);

        ids.push_back(id);

        if (id != draft[i]) {
            break;
        }
    }

    if (i == n_draft) {
        ids.push_back(llama_sampler_sample(smpl, ctx, (int32_t) i));
    }

    llama_sampler_reset(smpl);

    n_past = n_past_batch + 1 + (int) (ids.size() - 1);

    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, n_past, -1);

    return ids;
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SELF_SPEC_BIAS, print_usage)) {
        return 1;
    }

    if (params.sampling.temp > 0.0f) {
        LOG_ERR("%s: probability biasing requires greedy sampling, use --temp 0\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }

    // rejected draft tokens are dropped with llama_memory_seq_rm, so partial removal is mandatory
    const common_context_seq_rm_type seq_rm_type = common_context_can_seq_rm(ctx);

    if (seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART && seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
        LOG_ERR("%s: this context cannot remove partial sequences, draft reuse is not possible\n", __func__);
        return 1;
    }

    LOG_INF("%s: stream interval  = %d\n",     __func__, params.spec_bias_stream_interval);
    LOG_INF("%s: draft bias beta  = %.2f\n",   __func__, (double) params.spec_bias_draft_beta);
    LOG_INF("%s: target bias beta = %.2f\n",   __func__, (double) params.spec_bias_target_beta);
    LOG_INF("%s: draft reuse      = %s\n",     __func__, params.spec_bias_draft_reuse  ? "enabled" : "disabled");
    LOG_INF("%s: prompt cache     = %s\n",     __func__, params.spec_bias_prompt_cache ? "enabled" : "disabled");
    LOG_INF("%s: output mask k    = %d\n",     __func__, params.spec_bias_output_mask_k);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // one sampler verifies the reused draft, the other decodes past it
    //
    // both are wrapped in a chain: llama_sampler_sample reuses the chain candidate
    // buffer, a bare sampler would reallocate it on every call
    llama_sampler * bias_draft  = spec_bias_sampler_init(n_vocab, params.spec_bias_draft_beta);
    llama_sampler * bias_target = spec_bias_sampler_init(n_vocab, params.spec_bias_target_beta);

    llama_sampler * chain_draft  = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler * chain_target = llama_sampler_chain_init(llama_sampler_chain_default_params());

    llama_sampler_chain_add(chain_draft,  bias_draft);
    llama_sampler_chain_add(chain_target, bias_target);

    const llama_seq_id seq_id = 0;

    llama_batch batch = llama_batch_init(llama_n_batch(ctx), 0, 1);

    const int n_ctx = llama_n_ctx(ctx);

    llama_tokens cache_tokens; // prompt currently held in the kv cache

    // one json object per input line, holding every request and its answer
    std::ofstream f_json;

    if (!params.out_file.empty()) {
        f_json.open(params.out_file);

        if (!f_json) {
            LOG_ERR("%s: failed to open %s for writing\n", __func__, params.out_file.c_str());
            return 1;
        }
    }

    size_t n_prompt_reused = 0;
    size_t n_prompt_eval   = 0;
    size_t n_draft         = 0;
    size_t n_draft_reused  = 0;
    size_t n_decoded       = 0;
    size_t n_steps         = 0;

    int64_t t_prompt_us = 0;
    int64_t t_verify_us = 0;
    int64_t t_decode_us = 0;

    const std::vector<spec_bias_record> records = spec_bias_load(params);

    if (records.empty()) {
        LOG_ERR("%s: no input records\n", __func__);
        return 1;
    }

    for (const auto & rec_in : records) {
        const std::vector<std::string> & prefixes = rec_in.stream_ins;

        llama_tokens draft;    // previous output, reused as the draft
        std::string  prev_src; // previous source text, to detect a restart

        std::vector<std::string> stream_ins;
        std::vector<std::string> stream_outs;

        for (size_t i_pre = 0; i_pre < prefixes.size(); ++i_pre) {
            const std::string & src = prefixes[i_pre];

            // a source that does not extend the previous one invalidates the draft
            if (prev_src.size() > src.size() || src.compare(0, prev_src.size(), prev_src) != 0) {
                draft.clear();
            }
            prev_src = src;

            const llama_tokens prompt = common_tokenize(ctx, params.input_prefix + src + params.input_suffix, true, true);

            if ((int) prompt.size() + params.n_predict >= n_ctx) {
                LOG_ERR("%s: prompt plus prediction exceeds the context size\n", __func__);
                break;
            }

            if (prompt.size() > (size_t) llama_n_batch(ctx)) {
                LOG_ERR("%s: prompt of %zu tokens exceeds the batch size %u\n", __func__, prompt.size(), llama_n_batch(ctx));
                break;
            }

            size_t n_keep = 0;
            if (params.spec_bias_prompt_cache) {
                n_keep = spec_bias_prompt_reuse(ctx, seq_id, cache_tokens, prompt);
            } else {
                llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
            }
            cache_tokens = prompt;

            // hold back the last prompt token, it is decoded with the draft
            int n_past = (int) prompt.size() - 1;

            if ((int) n_keep < n_past) {
                const int64_t t_beg = ggml_time_us();

                common_batch_clear(batch);
                for (size_t i = n_keep; i + 1 < prompt.size(); ++i) {
                    common_batch_add(batch, prompt[i], (int) i, { seq_id }, false);
                }
                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("%s: failed to evaluate the prompt\n", __func__);
                    return 1;
                }

                t_prompt_us += ggml_time_us() - t_beg;
            }

            n_prompt_reused += n_keep;
            n_prompt_eval   += prompt.size() - n_keep;

            llama_token id_last = prompt.back();

            llama_tokens out;
            size_t n_accepted = 0;

            // id_last still holds the last prompt token, which must be decoded
            // but never emitted. the draft path replaces it with a real output.
            bool id_last_is_output = false;

            if (params.spec_bias_draft_reuse && !draft.empty()) {
                const int64_t t_beg = ggml_time_us();

                const llama_tokens ids = spec_bias_verify_draft(ctx, chain_draft, batch, seq_id, id_last, draft, n_past);

                t_verify_us += ggml_time_us() - t_beg;

                if (ids.empty()) {
                    return 1;
                }

                n_accepted = ids.size() - 1;

                out.insert(out.end(), ids.begin(), ids.end() - 1);

                id_last = ids.back();
                id_last_is_output = true;

                n_draft        += draft.size();
                n_draft_reused += n_accepted;
            }

            // the part of the draft the model rejected still biases the decode
            if (n_accepted < draft.size()) {
                spec_bias_sampler_set_seq(bias_target, draft, n_accepted);
            }

            bool eog = llama_vocab_is_eog(vocab, id_last);

            const int64_t t_beg_decode = ggml_time_us();

            while (!eog && n_past + 1 < n_ctx && (params.n_predict <= 0 || (int) out.size() < params.n_predict)) {
                common_batch_clear(batch);
                common_batch_add(batch, id_last, n_past++, { seq_id }, true);

                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("%s: failed to decode\n", __func__);
                    return 1;
                }

                if (id_last_is_output) {
                    out.push_back(id_last);
                    n_decoded++;
                }

                id_last = llama_sampler_sample(chain_target, ctx, -1);
                id_last_is_output = true;

                eog = llama_vocab_is_eog(vocab, id_last);
            }

            t_decode_us += ggml_time_us() - t_beg_decode;

            llama_sampler_reset(chain_target);

            // Arivazhagan et al: do not transmit the last k tokens of a partial
            // answer, they lean on source that has not arrived. The last answer
            // of a line is sent whole. What is not transmitted is not drafted
            // from either, so the model decides it again with more to go on.
            llama_tokens sent = out;
            if (params.spec_bias_output_mask_k > 0 && i_pre + 1 < prefixes.size()) {
                const size_t k = (size_t) params.spec_bias_output_mask_k;
                sent.resize(k < sent.size() ? sent.size() - k : 0);
            }

            std::string text;
            for (size_t i = 0; i < sent.size(); ++i) {
                text += common_token_to_piece(ctx, sent[i]);
            }

            LOG("%s\n", text.c_str());

            if (f_json.is_open()) {
                stream_ins.push_back(src);
                stream_outs.push_back(text);
            }
            LOG_DBG("%s: accepted %d/%d draft tokens\n", __func__, (int) n_accepted, (int) draft.size());

            draft = sent;

            n_steps++;
        }

        if (f_json.is_open() && !stream_outs.empty()) {
            nlohmann::ordered_json rec;

            rec["id"]           = rec_in.id;
            rec["source"]       = rec_in.source;
            rec["segmentation"] = rec_in.segmentation;
            rec["stream_ins"]   = stream_ins;
            rec["stream_outs"]  = stream_outs;

            f_json << rec.dump() << "\n";
        }
    }

    const size_t n_prompt = n_prompt_reused + n_prompt_eval;
    const size_t n_out    = n_decoded + n_draft_reused;

    LOG("\n");
    LOG_INF("%s: steps          = %zu\n", __func__, n_steps);
    LOG_INF("%s: prompt tokens  = %zu (%zu reused, %zu evaluated)\n", __func__, n_prompt, n_prompt_reused, n_prompt_eval);
    if (n_prompt > 0) {
        LOG_INF("%s:   reuse rate   = %.1f %%\n", __func__, 100.0 * n_prompt_reused / n_prompt);
    }
    LOG_INF("%s: output tokens  = %zu (%zu from draft, %zu decoded)\n", __func__, n_out, n_draft_reused, n_decoded);
    if (n_out > 0) {
        LOG_INF("%s:   from draft   = %.1f %%\n", __func__, 100.0 * n_draft_reused / n_out);
    }
    if (n_draft > 0) {
        LOG_INF("%s: draft accepted = %zu / %zu (%.1f %%)\n", __func__, n_draft_reused, n_draft, 100.0 * n_draft_reused / n_draft);
    } else {
        LOG_INF("%s: draft accepted = n/a (no draft was verified)\n", __func__);
    }
    LOG_INF("%s: prompt eval    = %.2f ms\n", __func__, t_prompt_us / 1000.0);
    LOG_INF("%s: draft verify   = %.2f ms\n", __func__, t_verify_us / 1000.0);
    LOG_INF("%s: decode         = %.2f ms\n", __func__, t_decode_us / 1000.0);
    if (t_decode_us + t_verify_us > 0) {
        LOG_INF("%s: output speed   = %.2f t/s\n", __func__, n_out / ((t_decode_us + t_verify_us) / 1e6));
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_batch_free(batch);

    llama_sampler_free(chain_target);
    llama_sampler_free(chain_draft);

    llama_backend_free();

    return 0;
}
