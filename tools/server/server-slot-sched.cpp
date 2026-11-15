#include "server-slot-sched.h"

#include "common.h"
#include "hash/hash.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <fstream>

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

//
// server_prompt_cache_disk
//

namespace fs = std::filesystem;

static const char * FILE_EXT     = ".kvc";
static const char * FILE_EXT_TMP = ".tmp";

struct server_cache_disk_header {
    uint32_t magic   = 0x3243564b; // "KVC2"
    uint32_t version = 1;

    uint64_t n_tokens      = 0;
    uint64_t n_hashes      = 0;
    uint64_t n_checkpoints = 0;

    // in bytes
    uint64_t size_tokens      = 0; // server_tokens::serialize()
    uint64_t size_main        = 0;
    uint64_t size_drft        = 0;
    uint64_t size_checkpoints = 0; // headers included

    server_tokens::prefix_hash hash = {};
};
static_assert(sizeof(server_cache_disk_header) == 96, "unexpected header size");

struct server_cache_disk_checkpoint {
    int64_t n_tokens = 0;
    int32_t pos_min  = 0;
    int32_t pos_max  = 0;

    // in bytes
    uint64_t size_tgt  = 0;
    uint64_t size_dft  = 0;
    uint64_t size_spec = 0;
};
static_assert(sizeof(server_cache_disk_checkpoint) == 40, "unexpected checkpoint header size");

static std::string make_name(const std::string & compat, const server_tokens::prefix_hash & hash) {
    static const char hex[] = "0123456789abcdef";

    // half of the hash is enough for a file name
    std::string res = compat + "-";
    for (size_t i = 0; i < hash.size() / 2; i++) {
        res += hex[hash[i] >> 4];
        res += hex[hash[i] & 0xf];
    }
    return res + FILE_EXT;
}

static void read_bytes(std::ifstream & in, void * data, size_t size) {
    in.read((char *) data, size);
    if (!in.good()) {
        throw std::runtime_error("unexpected end of file");
    }
}

static void write_bytes(std::ofstream & out, const void * data, size_t size) {
    out.write((const char *) data, size);
}

// reads the header and the hash chain, the stream is left at the start of the tokens
static server_cache_disk_header read_header(std::ifstream & in, const std::string & compat, server_prompt_cache_disk_entry & entry) {
    server_cache_disk_header header;
    read_bytes(in, &header, sizeof(header));

    if (header.magic != server_cache_disk_header().magic || header.version != server_cache_disk_header().version) {
        throw std::runtime_error("unsupported file version");
    }

    const uint64_t size_hashes = header.n_hashes * sizeof(server_tokens::prefix_hash);
    const uint64_t size_file   = sizeof(header) + size_hashes + header.size_tokens + header.size_main + header.size_drft + header.size_checkpoints;

    if (header.n_hashes != header.n_tokens / server_tokens::hash_block_size ||
            header.size_tokens % sizeof(llama_token) != 0 ||
            size_file != entry.size ||
            make_name(compat, header.hash) != entry.name) {
        throw std::runtime_error("invalid header");
    }

    entry.n_tokens = header.n_tokens;
    entry.hash     = header.hash;
    entry.hashes.resize(header.n_hashes);
    if (size_hashes > 0) {
        read_bytes(in, entry.hashes.data(), size_hashes);
    }

    return header;
}

server_prompt_cache_disk::server_prompt_cache_disk(const std::string & dir, const std::string & compat, size_t limit_size, bool has_mtmd)
        : dir(dir), compat(compat), limit_size(limit_size), has_mtmd(has_mtmd) {
    scan();
}

void server_prompt_cache_disk::scan() {
    size_t n_usable = 0;

    std::error_code ec;
    for (const auto & file : fs::directory_iterator(dir, ec)) {
        if (!file.is_regular_file(ec)) {
            continue;
        }

        const std::string name = file.path().filename().string();
        const bool is_compat   = string_starts_with(name, compat + "-");

        // left by a server that was killed while saving
        if (is_compat && string_ends_with(name, FILE_EXT_TMP)) {
            fs::remove(file.path(), ec);
            continue;
        }

        if (!string_ends_with(name, FILE_EXT)) {
            continue;
        }

        server_prompt_cache_disk_entry entry;
        entry.name   = name;
        entry.size   = file.file_size(ec);
        entry.t_used = file.last_write_time(ec);

        if (is_compat) {
            try {
                std::ifstream in(file.path(), std::ios::binary);
                read_header(in, compat, entry);
            } catch (const std::exception & e) {
                SRV_WRN("removing '%s': %s\n", name.c_str(), e.what());
                fs::remove(file.path(), ec);
                continue;
            }
            n_usable++;
        }

        entries.push_back(std::move(entry));
    }

    entries.sort([](const auto & a, const auto & b) {
        return a.t_used < b.t_used;
    });

    SRV_INF("disk prompt cache '%s': %zu prompts, %zu files of other models, %.3f MiB (limit: %.3f MiB)\n",
            dir.string().c_str(), n_usable, entries.size() - n_usable, size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
}

size_t server_prompt_cache_disk::size() const {
    size_t res = 0;

    for (const auto & entry : entries) {
        res += entry.size;
    }

    return res;
}

server_prompt_cache_disk::entry_list::iterator server_prompt_cache_disk::remove(entry_list::iterator it) {
    std::error_code ec;
    fs::remove(dir / it->name, ec);

    return entries.erase(it);
}

void server_prompt_cache_disk::touch(entry_list::iterator it) {
    std::error_code ec;
    it->t_used = fs::file_time_type::clock::now();
    fs::last_write_time(dir / it->name, it->t_used, ec);

    entries.splice(entries.end(), entries, it);
}

bool server_prompt_cache_disk::save(const server_prompt_cache_state & state) {
    const auto & tokens = state.prompt.tokens;

    if (tokens.empty() || state.data.main.empty()) {
        return false;
    }

    server_cache_disk_header header;
    header.hash = tokens.get_hash(tokens.size());

    const std::string name = make_name(compat, header.hash);

    for (auto it = entries.begin(); it != entries.end();) {
        if (it->name == name) {
            SRV_TRC(" - prompt is already on disk, '%s'\n", name.c_str());
            touch(it);
            return true;
        }

        if (0 < it->n_tokens && it->n_tokens < tokens.size() && tokens.get_hash(it->n_tokens) == it->hash) {
            SRV_TRC(" - removing obsolete prompt with length %zu from disk, '%s'\n", it->n_tokens, it->name.c_str());
            it = remove(it);
        } else {
            ++it;
        }
    }

    std::vector<char> tokens_data;
    try {
        tokens_data = tokens.serialize();
    } catch (const std::exception & e) {
        SRV_WRN("failed to serialize the prompt: %s\n", e.what());
        return false;
    }

    const auto & hashes = tokens.get_prefix_hashes();

    header.n_tokens      = tokens.size();
    header.n_hashes      = hashes.size();
    header.n_checkpoints = state.prompt.checkpoints.size();
    header.size_tokens   = tokens_data.size();
    header.size_main     = state.data.main.size();
    header.size_drft     = state.data.drft.size();

    for (const auto & ckpt : state.prompt.checkpoints) {
        header.size_checkpoints += sizeof(server_cache_disk_checkpoint) + ckpt->data_tgt.size() + ckpt->data_dft.size() + ckpt->data_spec.size();
    }

    const size_t size_hashes = hashes.size() * sizeof(server_tokens::prefix_hash);
    const size_t size_file   = sizeof(header) + size_hashes + header.size_tokens + header.size_main + header.size_drft + header.size_checkpoints;

    if (limit_size > 0) {
        if (size_file > limit_size) {
            SRV_WRN(" - prompt size %.3f MiB exceeds disk cache size limit %.3f MiB, skipping\n",
                    size_file / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
            return false;
        }

        while (!entries.empty() && size() + size_file > limit_size) {
            SRV_TRC(" - making room on disk, removing oldest file '%s' (size = %.3f MiB)\n",
                    entries.front().name.c_str(), entries.front().size / (1024.0 * 1024.0));
            remove(entries.begin());
        }
    }

    const fs::path path     = dir / name;
    const fs::path path_tmp = dir / (name + FILE_EXT_TMP);

    std::error_code ec;
    {
        std::ofstream out(path_tmp, std::ios::binary | std::ios::trunc);

        write_bytes(out, &header, sizeof(header));
        write_bytes(out, hashes.data(), size_hashes);
        write_bytes(out, tokens_data.data(), tokens_data.size());
        write_bytes(out, state.data.main.data(), state.data.main.size());
        write_bytes(out, state.data.drft.data(), state.data.drft.size());

        for (const auto & ckpt : state.prompt.checkpoints) {
            server_cache_disk_checkpoint ckpt_header;
            ckpt_header.n_tokens  = ckpt->n_tokens;
            ckpt_header.pos_min   = ckpt->pos_min;
            ckpt_header.pos_max   = ckpt->pos_max;
            ckpt_header.size_tgt  = ckpt->data_tgt.size();
            ckpt_header.size_dft  = ckpt->data_dft.size();
            ckpt_header.size_spec = ckpt->data_spec.size();

            write_bytes(out, &ckpt_header, sizeof(ckpt_header));
            write_bytes(out, ckpt->data_tgt.data(), ckpt->data_tgt.size());
            write_bytes(out, ckpt->data_dft.data(), ckpt->data_dft.size());
            write_bytes(out, ckpt->data_spec.data(), ckpt->data_spec.size());
        }

        out.flush();
        if (!out.good()) {
            SRV_ERR("failed to write '%s'\n", path_tmp.string().c_str());
            out.close();
            fs::remove(path_tmp, ec);
            return false;
        }
    }

    // a reader never sees a half written file
    fs::rename(path_tmp, path, ec);
    if (ec) {
        SRV_ERR("failed to rename '%s': %s\n", path_tmp.string().c_str(), ec.message().c_str());
        fs::remove(path_tmp, ec);
        return false;
    }

    server_prompt_cache_disk_entry entry;
    entry.name     = name;
    entry.size     = size_file;
    entry.t_used   = fs::last_write_time(path, ec);
    entry.n_tokens = tokens.size();
    entry.hash     = header.hash;
    entry.hashes   = hashes;
    entries.push_back(std::move(entry));

    SRV_INF("saved prompt with %zu tokens to disk, %.3f MiB, '%s'\n", tokens.size(), size_file / (1024.0 * 1024.0), name.c_str());

    return true;
}

bool server_prompt_cache_disk::load(const server_prompt_cache_disk_entry & entry, server_prompt_cache_state & state) {
    const auto it = std::find_if(entries.begin(), entries.end(), [&entry](const auto & cur) {
        return &cur == &entry;
    });
    GGML_ASSERT(it != entries.end());

    server_prompt_cache_state res;

    try {
        std::ifstream in(dir / it->name, std::ios::binary);

        server_prompt_cache_disk_entry cur;
        cur.name = it->name;
        cur.size = it->size;
        const auto header = read_header(in, compat, cur);

        llama_tokens tokens_data(header.size_tokens / sizeof(llama_token));
        read_bytes(in, tokens_data.data(), header.size_tokens);

        res.prompt.tokens = server_tokens::deserialize(tokens_data, has_mtmd);

        if (res.prompt.tokens.size() != it->n_tokens || res.prompt.tokens.get_hash(it->n_tokens) != it->hash) {
            throw std::runtime_error("the tokens do not match the hash");
        }

        res.data.main.resize(header.size_main);
        res.data.drft.resize(header.size_drft);
        read_bytes(in, res.data.main.data(), header.size_main);
        read_bytes(in, res.data.drft.data(), header.size_drft);

        uint64_t size_checkpoints = 0;
        for (uint64_t i = 0; i < header.n_checkpoints; i++) {
            server_cache_disk_checkpoint ckpt_header;
            read_bytes(in, &ckpt_header, sizeof(ckpt_header));

            size_checkpoints += sizeof(ckpt_header) + ckpt_header.size_tgt + ckpt_header.size_dft + ckpt_header.size_spec;
            if (size_checkpoints > header.size_checkpoints) {
                throw std::runtime_error("invalid checkpoint size");
            }

            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            ckpt->n_tokens = ckpt_header.n_tokens;
            ckpt->pos_min  = ckpt_header.pos_min;
            ckpt->pos_max  = ckpt_header.pos_max;
            ckpt->data_tgt.resize(ckpt_header.size_tgt);
            ckpt->data_dft.resize(ckpt_header.size_dft);
            ckpt->data_spec.resize(ckpt_header.size_spec);
            read_bytes(in, ckpt->data_tgt.data(), ckpt_header.size_tgt);
            read_bytes(in, ckpt->data_dft.data(), ckpt_header.size_dft);
            read_bytes(in, ckpt->data_spec.data(), ckpt_header.size_spec);

            res.prompt.checkpoints.push_back(std::move(ckpt));
        }
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for '%s': %s\n", it->name.c_str(), e.what());
        return false;
    } catch (const std::exception & e) {
        SRV_WRN("removing '%s': %s\n", it->name.c_str(), e.what());
        remove(it);
        return false;
    }

    SRV_INF("loaded prompt with %zu tokens from disk, %.3f MiB, '%s'\n", it->n_tokens, it->size / (1024.0 * 1024.0), it->name.c_str());

    touch(it);

    state = std::move(res);

    return true;
}

template <typename T>
static void compat_add(std::string & buf, const T & value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    buf.append((const char *) &value, sizeof(value));
}

static void compat_add(std::string & buf, const std::string & value) {
    compat_add(buf, (uint64_t) value.size());
    buf += value;
}

// a file is known by its path, size and time, reading a model to hash it takes too long
static void compat_add_file(std::string & buf, const std::string & path) {
    std::error_code ec;
    compat_add(buf, path);
    compat_add(buf, (uint64_t) (path.empty() ? 0 : fs::file_size(path, ec)));
    compat_add(buf, (int64_t)  (path.empty() ? 0 : fs::last_write_time(path, ec).time_since_epoch().count()));
}

std::string server_prompt_cache_disk_compat(const common_params & params) {
    std::string buf;

    compat_add(buf, (uint32_t) LLAMA_STATE_SEQ_VERSION);
    compat_add(buf, (uint64_t) server_tokens::hash_block_size);

    compat_add_file(buf, params.model.path);
    compat_add_file(buf, params.mmproj.path);
    compat_add_file(buf, params.speculative.draft.mparams.path);

    for (const auto & lora : params.lora_adapters) {
        compat_add_file(buf, lora.path);
        compat_add(buf, lora.scale);
    }

    for (const auto & cvec : params.control_vectors) {
        compat_add_file(buf, cvec.fname);
        compat_add(buf, cvec.strength);
    }
    compat_add(buf, params.control_vector_layer_start);
    compat_add(buf, params.control_vector_layer_end);

    compat_add(buf, (int32_t) params.cache_type_k);
    compat_add(buf, (int32_t) params.cache_type_v);
    compat_add(buf, (uint8_t) params.swa_full);

    compat_add(buf, params.rope_freq_base);
    compat_add(buf, params.rope_freq_scale);
    compat_add(buf, (int32_t) params.rope_scaling_type);
    compat_add(buf, params.yarn_ext_factor);
    compat_add(buf, params.yarn_attn_factor);
    compat_add(buf, params.yarn_beta_fast);
    compat_add(buf, params.yarn_beta_slow);
    compat_add(buf, params.yarn_orig_ctx);

    return hash_sha256_hex(buf.data(), buf.size()).substr(0, 16);
}

//
// server_prompt_cache
//

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            pop_oldest();
        }
    }

    common_state_data state_data_tgt;
    common_state_data state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens       =*/ prompt.tokens.clone(),
            /*.checkpoints  =*/ prompt.checkpoints,
            /*.lora_changed =*/ prompt.lora_changed,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

void server_prompt_cache::save_to_disk(const server_prompt_cache_state & state) {
    if (!disk || state.prompt.lora_changed) {
        return;
    }

    disk->save(state);
}

void server_prompt_cache::flush_to_disk() {
    for (const auto & state : states) {
        save_to_disk(state);
    }
}

void server_prompt_cache::pop_oldest() {
    save_to_disk(states.front());
    states.pop_front();
}

void server_prompt_cache::load_from_disk(const server_tokens & tokens_new, size_t lcp_slot, int32_t n_ctx_slot) {
    size_t lcp_best = lcp_slot;

    for (const auto & state : states) {
        lcp_best = std::max(lcp_best, state.prompt.tokens.get_common_prefix(tokens_new));
    }

    const server_prompt_cache_disk_entry * entry_best = nullptr;

    for (const auto & entry : disk->get_entries()) {
        if (entry.n_tokens == 0 || entry.n_tokens > (size_t) n_ctx_slot) {
            continue;
        }

        // only the hashes are known here, so this is a lower bound
        const size_t lcp_cur = tokens_new.get_common_prefix_lower(entry.hashes);

        // don't trash large prompts
        if (lcp_cur < 0.25f * entry.n_tokens) {
            continue;
        }

        if (lcp_cur > lcp_best) {
            lcp_best   = lcp_cur;
            entry_best = &entry;
        }
    }

    if (entry_best == nullptr) {
        return;
    }

    SRV_TRC(" - found better prompt on disk with length %zu, lcp >= %zu\n", entry_best->n_tokens, lcp_best);

    server_prompt_cache_state state;
    if (disk->load(*entry_best, state)) {
        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__\n");
        states.push_back(std::move(state));
    }
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, int32_t n_ctx_slot) {
    int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    if (disk) {
        load_from_disk(tokens_new, lcp_best, n_ctx_slot);
    }

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp_best    = lcp_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

        // the whole cached prompt is a prefix of the new one
        if (lcp_best == (int) it_best->prompt.tokens.size()) {
            SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            pop_oldest();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            pop_oldest();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const float f_sim_cur = float(lcp_len) / task.tokens.size();

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size();

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    update_cache = update_cache && !task.is_stateless();

    return { ret->id, update_cache };
}
