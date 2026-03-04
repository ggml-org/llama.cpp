#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <fstream>
#include <iostream> // TODO: remove me

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s --model ./models/bge-base-en-v1.5-f16.gguf --top-k 3 --context-file README.md --context-file License --chunk-size 100 --chunk-separator .\n", argv[0]);
    LOG("\n");
}

struct chunk {
    // filename
    std::string filename;
    // original file position
    size_t filepos;
    // original text data
    std::string textdata;
    // tokenized text data
    std::vector<llama_token> tokens;
    // embedding
    std::vector<float> embedding;
};

// chunk file data to chunks of size >= chunk_size
// chunk_separator is the separator between chunks
static std::vector<chunk> chunk_file(const std::string & filename, int chunk_size, const std::string & chunk_separator) {
    std::vector<chunk> chunks;
    std::ifstream f(filename.c_str());

    if (!f.is_open()) {
        LOG_ERR("could not open file %s\n", filename.c_str());
        return chunks;
    }

    chunk current_chunk;
    char buffer[1024];
    int64_t filepos = 0;
    std::string current;
    while (f.read(buffer, 1024)) {
        current += std::string(buffer, f.gcount());
        size_t pos;
        while ((pos = current.find(chunk_separator)) != std::string::npos) {
            current_chunk.textdata += current.substr(0, pos + chunk_separator.size());
            if ((int) current_chunk.textdata.size() > chunk_size) {
                // save chunk
                current_chunk.filepos = filepos;
                current_chunk.filename = filename;
                chunks.push_back(current_chunk);
                // update filepos
                filepos += (int) current_chunk.textdata.size();
                // reset current_chunk
                current_chunk = chunk();
            }
            current = current.substr(pos + chunk_separator.size());
        }

    }
    // add leftover data to last chunk
    if (current_chunk.textdata.size() > 0) {
        if (chunks.empty()) {
            current_chunk.filepos = filepos;
            current_chunk.filename = filename;
            chunks.push_back(current_chunk);
        } else {
            chunks.back().textdata += current_chunk.textdata;
        }
    }
    f.close();
    return chunks;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
}

static void batch_process(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), false);

    // run model
    LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        LOG_ERR("%s : failed to process\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                LOG_ERR("%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        common_embd_normalize(embd, out, n_embd, 2);
    }
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_RETRIEVAL, print_usage)) {
        return 1;
    }

    common_init();

    // For BERT models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
    params.embedding = true;

    if (params.chunk_size <= 0) {
        LOG_ERR("chunk_size must be positive\n");
        return 1;
    }
    if (params.context_files.empty()) {
        LOG_ERR("context_files must be specified\n");
        return 1;
    }

    LOG_INF("processing files:\n");
    for (auto & context_file : params.context_files) {
        LOG_INF("%s\n", context_file.c_str());
    }

    std::vector<chunk> chunks;
    for (auto & context_file : params.context_files) {
        std::vector<chunk> file_chunk = chunk_file(context_file, params.chunk_size, params.chunk_separator);
        chunks.insert(chunks.end(), file_chunk.begin(), file_chunk.end());
    }
    LOG_INF("Number of chunks: %zu\n", chunks.size());

    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        LOG_ERR("%s: pooling type NONE not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    // max batch size
    const uint64_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    // tokenize the prompts and trim
    for (auto & chunk : chunks) {
        auto inp = common_tokenize(ctx, chunk.textdata, true, false);
        if (inp.size() > n_batch) {
            LOG_ERR("%s: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        // add eos if not present
        if (llama_vocab_eos(vocab) >= 0 && (inp.empty() || inp.back() != llama_vocab_eos(vocab))) {
            inp.push_back(llama_vocab_eos(vocab));
        }
        chunk.tokens = inp;
    }

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) chunks.size(); i++) {
            LOG_INF("%s: prompt %d: '%s'\n", __func__, i, chunks[i].textdata.c_str());
            LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, chunks[i].tokens.size());
            for (int j = 0; j < (int) chunks[i].tokens.size(); j++) {
                LOG_INF("%6d -> '%s'\n", chunks[i].tokens[j], common_token_to_piece(ctx, chunks[i].tokens[j]).c_str());
            }
            LOG_INF("\n\n");
        }
    }

    // initialize batch
    const int n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd_out = llama_model_n_embd_out(model);
    std::vector<float> embeddings(n_chunks * n_embd_out, 0);
    float * emb = embeddings.data();

    // break into batches
    unsigned int p = 0; // number of prompts processed already
    unsigned int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_chunks; k++) {
        // clamp to n_batch tokens
        auto & inp = chunks[k].tokens;

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch || s >= llama_n_seq_max(ctx)) {
            float * out = emb + p * n_embd_out;
            batch_process(ctx, batch, out, s, n_embd_out);
            common_batch_clear(batch);
            p += s;
            s = 0;
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + p * n_embd_out;
    batch_process(ctx, batch, out, s, n_embd_out);

    // save embeddings to chunks
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embedding = std::vector<float>(emb + i * n_embd_out, emb + (i + 1) * n_embd_out);
        // clear tokens as they are no longer needed
        chunks[i].tokens.clear();
    }

    struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);

    // start loop, receive query and return top k similar chunks based on cosine similarity
    std::string query;
    while (true) {
        LOG("Enter query: ");
        std::getline(std::cin, query);
        std::vector<int32_t> query_tokens = common_tokenize(ctx, query, true);

        batch_add_seq(query_batch, query_tokens, 0);

        std::vector<float> query_emb(n_embd_out, 0);
        batch_process(ctx, query_batch, query_emb.data(), 1, n_embd_out);

        common_batch_clear(query_batch);

        // compute cosine similarities
        {
            std::vector<std::pair<int, float>> similarities;
            for (int i = 0; i < n_chunks; i++) {
                float sim = common_embd_similarity_cos(chunks[i].embedding.data(), query_emb.data(), n_embd_out);
                similarities.push_back(std::make_pair(i, sim));
            }

            // sort similarities
            std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
                return a.second > b.second;
            });

            LOG("Top %d similar chunks:\n", params.sampling.top_k);
            for (int i = 0; i < std::min(params.sampling.top_k, (int) chunks.size()); i++) {
                LOG("filename: %s\n", chunks[similarities[i].first].filename.c_str());
                LOG("filepos: %lld\n", (long long int) chunks[similarities[i].first].filepos);
                LOG("similarity: %f\n", similarities[i].second);
                LOG("textdata:\n%s\n", chunks[similarities[i].first].textdata.c_str());
                LOG("--------------------\n");
            }
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    // clean up
    llama_batch_free(query_batch);
    llama_backend_free();
}
