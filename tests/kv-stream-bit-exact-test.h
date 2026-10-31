#pragma once

#include "testing.h"
#include "../common/speculative.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-kv-stream-model.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <exception>
#include <iterator>
#include <sstream>

struct kv_stream_exact_file_free {
    void operator()(FILE * file) const { if (file) std::fclose(file); }
};

// Compare encoded values, including signed zero; numerical tolerances do not apply.
static bool kv_stream_bytes_equal(const void * expected, size_t expected_bytes, const void * actual, size_t actual_bytes) {
    return expected_bytes == actual_bytes && (!actual_bytes ||
        (expected && actual && std::memcmp(expected, actual, actual_bytes) == 0));
}

// Keep the control on disk so full recurrent snapshots do not accumulate in RAM.
class kv_stream_bit_oracle {
public:
    using reader = std::function<void(size_t, uint8_t *, size_t)>;
    static constexpr size_t chunk_bytes = 1048576;
    kv_stream_bit_oracle() : file(std::tmpfile()), expected(chunk_bytes), actual(chunk_bytes) {
        if (!file) throw std::runtime_error("cannot create bit-exact control file");
    }

    void compare() {
        if (comparing || std::fflush(file.get()) || std::fseek(file.get(), 0, SEEK_SET))
            throw std::runtime_error("cannot rewind bit-exact control");
        comparing = true;
    }

    void data(const std::string & label, size_t count, const reader & source) {
        if (comparing) {
            if (cursor == entries.size() || entries[cursor].label != label || entries[cursor].bytes != count)
                throw std::runtime_error("bit-exact record order/size differs: " + label);
            ++cursor;
        } else {
            if (count > (uint64_t(32) << 30) - total)
                throw std::runtime_error("bit-exact control exceeds 32 GiB; reduce rounds");
            entries.push_back({label, count});
            total += count;
        }
        for (size_t offset = 0; offset < count;) {
            const size_t size = std::min(chunk_bytes, count-offset);
            source(offset, actual.data(), size);
            if (comparing) {
                if (std::fread(expected.data(), 1, size, file.get()) != size)
                    throw std::runtime_error("truncated bit-exact control: " + label);
                if (!kv_stream_bytes_equal(expected.data(), size, actual.data(), size)) {
                    const auto mismatch = std::mismatch(expected.begin(), expected.begin()+size, actual.begin());
                    const size_t byte = size_t(mismatch.first-expected.begin());
                    throw std::runtime_error("bit-exact mismatch: " + label + " byte=" + std::to_string(offset+byte) +
                        " expected=" + std::to_string(expected[byte]) + " actual=" + std::to_string(actual[byte]));
                }
            } else if (std::fwrite(actual.data(), 1, size, file.get()) != size) {
                throw std::runtime_error("cannot write bit-exact control: " + label);
            }
            offset += size;
        }
    }

    void bytes(const std::string & label, const void * source, size_t count) {
        if (count && !source) throw std::runtime_error("null bit-exact source");
        data(label, count, [&](size_t offset, uint8_t * dest, size_t size) {
            if (size) std::memcpy(dest, static_cast<const uint8_t *>(source) + offset, size);
        });
    }

    void tensor(const std::string & label, ggml_tensor * tensor, bool finite = false) {
        if (!tensor || !ggml_is_contiguous(tensor)) throw std::runtime_error("invalid exact tensor: " + label);
        const std::array<int64_t, 5> shape{int64_t(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]};
        bytes(label + "/shape", shape.data(), sizeof(shape));
        data(label, ggml_nbytes(tensor), [&](size_t offset, uint8_t * dest, size_t count) {
            ggml_backend_tensor_get(tensor, dest, offset, count);
            if (finite) {
                if (tensor->type != GGML_TYPE_F32 || count%sizeof(float)) throw std::runtime_error("expected FP32 tensor");
                for (size_t i = 0; i < count; i += sizeof(float)) {
                    float value;
                    std::memcpy(&value, dest+i, sizeof(value));
                    if (!std::isfinite(value)) throw std::runtime_error("nonfinite exact tensor: " + label);
                }
            }
        });
    }

    void finish() const {
        if (!comparing || cursor != entries.size()) throw std::runtime_error("incomplete bit-exact comparison");
    }

    size_t records() const { return entries.size(); }
    uint64_t byte_count() const { return total; }

    void save_control(FILE * dest, const std::array<uint64_t,16> & profile) {
        if (!dest || comparing || entries.size()>100000 || std::fflush(file.get()) || std::fseek(file.get(),0,SEEK_SET))
            throw std::runtime_error("cannot export bit-exact control");
        write(dest,"KVBIT001",8);
        for (auto word:profile) write_word(dest,word);
        write_word(dest,entries.size());write_word(dest,total);
        size_t label_bytes=0;
        for (const auto & item:entries) {
            if (item.label.size()>4096 || item.label.size()>(size_t(16)<<20)-label_bytes)
                throw std::runtime_error("bit-exact manifest is too large");
            label_bytes+=item.label.size();
            write_word(dest,item.label.size());write_word(dest,item.bytes);
            write(dest,item.label.data(),item.label.size());
        }
        for (uint64_t offset=0;offset<total;) {
            const auto count=size_t(std::min(uint64_t(chunk_bytes),total-offset));
            read(file.get(),actual.data(),count);write(dest,actual.data(),count);offset+=count;
        }
        if (std::fflush(dest)) throw std::runtime_error("cannot flush bit-exact control");
    }

    void load_control(FILE * source, const std::array<uint64_t,16> & profile) {
        if (!source || comparing) throw std::runtime_error("cannot import bit-exact control");
        char magic[8];read(source,magic,sizeof(magic));
        if (std::memcmp(magic,"KVBIT001",sizeof(magic))) throw std::runtime_error("invalid bit-exact control version");
        for (auto word:profile) {
            if (read_word(source)!=word) throw std::runtime_error("bit-exact control profile differs");
        }
        const auto count=read_word(source),bytes=read_word(source);
        if (count>100000 || bytes>(uint64_t(32)<<30)) throw std::runtime_error("bit-exact control exceeds limits");
        std::vector<entry> manifest;
        uint64_t sum=0,label_bytes=0;
        for (uint64_t i=0;i<count;++i) {
            const auto label_size=read_word(source),size=read_word(source);
            if (label_size>4096 || label_size>(uint64_t(16)<<20)-label_bytes || size>bytes-sum || size>SIZE_MAX)
                throw std::runtime_error("invalid bit-exact control manifest");
            std::string label(size_t(label_size),'\0');read(source,label.data(),label.size());
            manifest.push_back({std::move(label),size_t(size)});sum+=size;label_bytes+=label_size;
        }
        if (sum!=bytes) throw std::runtime_error("bit-exact control extent differs");
        std::unique_ptr<FILE,kv_stream_exact_file_free> imported(std::tmpfile());
        if (!imported) throw std::runtime_error("cannot create imported bit-exact control");
        for (uint64_t offset=0;offset<bytes;) {
            const auto size=size_t(std::min(uint64_t(chunk_bytes),bytes-offset));
            read(source,actual.data(),size);write(imported.get(),actual.data(),size);offset+=size;
        }
        if (std::fgetc(source)!=EOF || std::ferror(source) || std::fflush(imported.get()))
            throw std::runtime_error("trailing data or I/O error in bit-exact control");
        file=std::move(imported);entries=std::move(manifest);total=bytes;cursor=0;
    }

private:
    static void read(FILE * source, void * dest, size_t bytes) {
        if (std::fread(dest,1,bytes,source)!=bytes) throw std::runtime_error("truncated bit-exact control");
    }
    static void write(FILE * dest, const void * source, size_t bytes) {
        if (std::fwrite(source,1,bytes,dest)!=bytes) throw std::runtime_error("cannot write bit-exact control");
    }
    static uint64_t read_word(FILE * source) {
        std::array<uint8_t,8> bytes;read(source,bytes.data(),bytes.size());
        uint64_t result=0;
        for (size_t i=0;i<bytes.size();++i) result|=uint64_t(bytes[i])<<(8*i);
        return result;
    }
    static void write_word(FILE * dest, uint64_t value) {
        std::array<uint8_t,8> bytes;
        for (size_t i=0;i<bytes.size();++i) bytes[i]=uint8_t(value>>(8*i));
        write(dest,bytes.data(),bytes.size());
    }
    struct entry { std::string label; size_t bytes; };
    std::unique_ptr<FILE, kv_stream_exact_file_free> file;
    std::vector<entry> entries;
    std::vector<uint8_t> expected, actual;
    size_t cursor = 0;
    uint64_t total = 0;
    bool comparing = false;
};

// Canonical serialization must not depend on write/write_tensor chunk boundaries.
struct kv_stream_exact_state : llama_io_write_i {
    std::unique_ptr<FILE, kv_stream_exact_file_free> file{std::tmpfile()};
    std::vector<uint8_t> scratch = std::vector<uint8_t>(kv_stream_bit_oracle::chunk_bytes);
    size_t count = 0;
    kv_stream_exact_state() {
        if (!file) throw std::runtime_error("cannot create exact state file");
    }
    void write(const void * source, size_t bytes) override {
        if ((bytes && !source) || bytes > SIZE_MAX-count || std::fwrite(source, 1, bytes, file.get()) != bytes)
            throw std::runtime_error("cannot write exact state");
        count += bytes;
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t bytes) override {
        if (!tensor || offset > ggml_nbytes(tensor) || bytes > ggml_nbytes(tensor)-offset)
            throw std::runtime_error("invalid exact state tensor range");
        for (size_t done = 0; done < bytes;) {
            const size_t size = std::min(scratch.size(), bytes-done);
            ggml_backend_tensor_get(tensor, scratch.data(), offset+done, size);
            write(scratch.data(), size);
            done += size;
        }
    }
    size_t n_bytes() override { return count; }
    void compare_with(kv_stream_bit_oracle & oracle, const std::string & label) {
        if (std::fflush(file.get()) || std::fseek(file.get(), 0, SEEK_SET)) throw std::runtime_error("cannot rewind exact state");
        oracle.data(label, count, [&](size_t, uint8_t * dest, size_t bytes) {
            if (std::fread(dest, 1, bytes, file.get()) != bytes) throw std::runtime_error("truncated exact state");
        });
    }
};

struct kv_stream_exact_trace {
    kv_stream_bit_oracle & oracle;
    std::string role, phase;
    size_t outputs = 0;
    std::exception_ptr failure;
    std::function<void()> prediction_attention;
    kv_stream_exact_trace(kv_stream_bit_oracle & oracle, std::string role) : oracle(oracle), role(std::move(role)) {}
    static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
        auto & trace = *static_cast<kv_stream_exact_trace *>(opaque);
        const bool attention = trace.prediction_attention && tensor && tensor->op == GGML_OP_FLASH_ATTN_EXT &&
            trace.phase.find("/predict") != std::string::npos;
        if (!tensor || trace.phase.empty() || trace.failure ||
                (!attention && std::strcmp(tensor->name, "result_output") && std::strcmp(tensor->name, "h_nextn") &&
                 (trace.role != "target" || std::strcmp(tensor->name, "result_norm")))) return false;
        if (ask) return true;
        try {
            if (attention) {
                trace.prediction_attention();
                return true;
            }
            trace.oracle.tensor(trace.phase + "/" + trace.role + "/" + tensor->name, tensor, true);
            if (!std::strcmp(tensor->name, "result_output")) ++trace.outputs;
        } catch (...) {
            trace.failure = std::current_exception();
            return false;
        }
        return true;
    }
    void check() const { if (failure) std::rethrow_exception(failure); }
};

// Read valid KV rows in logical order, not allocation order or padded capacity.
static void kv_stream_exact_kv(kv_stream_bit_oracle & oracle, const std::string & label,
        llama_kv_cache * kv, size_t first, size_t count) {
    const auto & cells = kv->get_cells(0);
    std::vector<uint32_t> slots(count, UINT32_MAX);
    for (const auto & item : cells.seq_pos_get(0)) {
        if (item.first >= 0 && size_t(item.first) >= first && size_t(item.first)-first < count)
            slots[size_t(item.first)-first] = item.second;
    }
    if (std::find(slots.begin(), slots.end(), UINT32_MAX) != slots.end()) throw std::runtime_error("missing logical KV row: " + label);
    const auto layers = kv->get_layer_ids();
    oracle.bytes(label + "/layers", layers.data(), layers.size()*sizeof(layers[0]));
    for (uint32_t layer : layers) for (bool value : {false, true}) {
        auto * tensor = value ? kv->get_v_storage(layer) : kv->get_k_storage(layer);
        if (!tensor || tensor->ne[2] != 1) throw std::runtime_error("invalid serial KV storage");
        const size_t row = ggml_row_size(tensor->type, tensor->ne[0]);
        if (!row || tensor->nb[1] != row || count > SIZE_MAX/row) throw std::runtime_error("invalid packed KV row");
        const auto tag = label + "/L" + std::to_string(layer) + (value ? "/V" : "/K");
        const std::array<uint64_t, 5> geometry{uint64_t(tensor->type), uint64_t(tensor->ne[0]), row, first, count};
        oracle.bytes(tag + "/shape", geometry.data(), sizeof(geometry));
        oracle.data(tag, count*row, [&](size_t offset, uint8_t * dest, size_t bytes) {
            while (bytes) {
                const size_t token = offset/row, within = offset%row;
                size_t run = 1;
                while (token+run < slots.size() && slots[token+run] == slots[token]+run) ++run;
                const size_t size = std::min(bytes, run*row-within);
                if (size_t(slots[token]) > (SIZE_MAX-within)/row) throw std::runtime_error("KV row offset overflow");
                const size_t physical = size_t(slots[token])*row+within;
                if (physical > ggml_nbytes(tensor) || size > ggml_nbytes(tensor)-physical)
                    throw std::runtime_error("KV row exceeds storage: " + tag);
                ggml_backend_tensor_get(tensor, dest, physical, size);
                offset += size; dest += size; bytes -= size;
            }
        });
    }
}

struct kv_stream_exact_config {
    size_t prefill = 16381, rounds = 8, pool_mib = 32;
    size_t batch_rows = 256, length = 0;
    ggml_type key = GGML_TYPE_Q8_0, value = GGML_TYPE_Q4_0;
    std::filesystem::path control_export, control_import;
    std::filesystem::path corpus = std::filesystem::path(__FILE__).parent_path().parent_path()/"benchmarks/data/online-articles-262144-words.txt";
};

// Test fixtures use the same public cache names without interpreting weight formats as KV types.
static ggml_type kv_stream_exact_cache_type(const std::string & name) {
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0,
            GGML_TYPE_Q5_1, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL}) {
        if (name == ggml_type_name(type)) return type;
    }
    throw std::runtime_error("unknown test KV cache type: " + name);
}

static kv_stream_exact_config kv_stream_exact_arguments(int argc, char ** argv) {
    kv_stream_exact_config cfg;
    for (int i = 3; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--mtp-bit-exact") continue;
        if (name != "--exact-prefill" && name != "--exact-rounds" && name != "--exact-pool-mib" && name != "--exact-prefill-file" &&
                name != "--test-cache-k" && name != "--test-cache-v" &&
                name != "--exact-batch-size" && name != "--exact-length" &&
                name != "--exact-control-export" && name != "--exact-control-import")
            throw std::runtime_error("unknown bit-exact test option: " + name);
        if (++i == argc) throw std::runtime_error("missing bit-exact option value");
        if (name == "--exact-prefill-file") { cfg.corpus = argv[i]; continue; }
        if (name == "--exact-control-export" || name == "--exact-control-import") {
            if (!*argv[i]) throw std::runtime_error("empty exact control path");
            (name == "--exact-control-export" ? cfg.control_export : cfg.control_import)=argv[i];
            continue;
        }
        if (name == "--test-cache-k") { cfg.key = kv_stream_exact_cache_type(argv[i]); continue; }
        if (name == "--test-cache-v") { cfg.value = kv_stream_exact_cache_type(argv[i]); continue; }
        size_t parsed = 0;
        const std::string input = argv[i];
        const auto value = std::stoull(input, &parsed);
        const size_t maximum = name == "--exact-prefill" ? 32768 : name == "--exact-rounds" ? 32 :
            name == "--exact-batch-size" ? 256 : name == "--exact-length" ? 3 : 1024;
        if (input.empty() || input[0] == '-' || parsed != input.size() || !value || value > maximum)
            throw std::runtime_error("invalid bit-exact option value: " + name);
        if (name == "--exact-prefill") cfg.prefill = size_t(value);
        else if (name == "--exact-rounds") cfg.rounds = size_t(value);
        else if (name == "--exact-batch-size") cfg.batch_rows = size_t(value);
        else if (name == "--exact-length") cfg.length = size_t(value);
        else cfg.pool_mib = size_t(value);
    }
    if (cfg.prefill < 512 || cfg.rounds < 4) throw std::runtime_error("bit-exact test needs at least 512 prefill tokens and four rounds");
    if (cfg.batch_rows < 5) throw std::runtime_error("bit-exact test needs more than four batch rows for recurrent rollback");
    if ((!cfg.control_export.empty() && !cfg.control_import.empty()) ||
            ((!cfg.control_export.empty() || !cfg.control_import.empty()) && !cfg.length))
        throw std::runtime_error("exact control needs one direction and one explicit MTP length");
    return cfg;
}

// Bound corpus allocation before reading; reject nonseekable or changed input.
static std::string kv_stream_exact_text(std::istream & input) {
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size <= 0 || size > std::streamoff(64*1048576)) throw std::runtime_error("bit-exact corpus must be 1 byte to 64 MiB");
    input.seekg(0, std::ios::beg);
    std::string text(size_t(size), '\0');
    if (!input.read(text.data(), std::streamsize(text.size())) || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("cannot read stable bit-exact corpus");
    return text;
}

// Use production MTP calls, but control acceptance to exercise every rollback depth.
static void test_kv_stream_mtp_bit_exact(testing & t, llama_model * model, const kv_stream_exact_config & cfg) {
    if (llama_model_n_layer_nextn(model) != 1) throw std::runtime_error("bit-exact MTP fixture requires one embedded Qwen MTP layer");
    std::ifstream input(cfg.corpus, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read bit-exact corpus: " + cfg.corpus.string());
    const auto text = kv_stream_exact_text(input);
    auto prompt = common_tokenize(llama_model_get_vocab(model), text, true, false);
    if (prompt.size() < cfg.prefill) throw std::runtime_error("bit-exact corpus is too short");
    prompt.resize(cfg.prefill);
    const size_t vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
    bool observed_mtp_streaming = false;
    for (size_t length : {1u, 2u, 3u}) {
        if (cfg.length && length != cfg.length) continue;
        t.test("strict_mtp_length_" + std::to_string(length), [&](testing & t) {
        kv_stream_bit_oracle oracle;
        const std::array<uint64_t,16> profile{0x4d545031,cfg.prefill,cfg.rounds,cfg.pool_mib,cfg.batch_rows,length,
            uint64_t(cfg.key),uint64_t(cfg.value),vocabulary,uint64_t(llama_model_n_embd(model)),
            uint64_t(llama_model_n_layer(model)),uint64_t(llama_model_n_layer_nextn(model)),
            (cfg.prefill+cfg.rounds*4+255)/256*256,3,1,0};
        if (!cfg.control_import.empty()) {
            std::unique_ptr<FILE,kv_stream_exact_file_free> source(std::fopen(cfg.control_import.string().c_str(),"rb"));
            if (!source) throw std::runtime_error("cannot open native MTP control");
            oracle.load_control(source.get(),profile);
        }
        for (bool streaming : {false, true}) {
            if (!streaming && !cfg.control_import.empty()) continue;
            if (streaming) oracle.compare();
            llama_tokens history = prompt;
            kv_stream_exact_trace target_trace{oracle, "target"}, draft_trace{oracle, "draft"};
            auto p = llama_context_default_params();
            p.n_ctx = uint32_t(((cfg.prefill + cfg.rounds*4 + 255)/256)*256);
            p.n_batch = p.n_ubatch = uint32_t(cfg.batch_rows); p.n_seq_max = 1; p.n_rs_seq = 3;
            p.n_outputs_max = p.n_outputs_max_per_seq = 256;
            p.n_threads = p.n_threads_batch = 8;
            p.type_k = cfg.key; p.type_v = cfg.value;
            p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            p.kv_stream_pool_bytes = streaming ? cfg.pool_mib*1048576 : 0;
            p.kv_stream_auxiliary_layers = streaming ? 1 : 0;
            p.cb_eval = kv_stream_exact_trace::callback; p.cb_eval_user_data = &target_trace;
            std::unique_ptr<llama_context, decltype(&llama_free)> target(llama_init_from_model(model, p), &llama_free);
            if (!target) throw std::runtime_error("cannot construct exact target");
            llama_set_embeddings_nextn(target.get(), true, false);
            auto d = p;
            d.ctx_type = LLAMA_CONTEXT_TYPE_MTP; d.ctx_other = target.get(); d.n_rs_seq = 0;
            d.kv_stream_auxiliary_layers = 0; d.kv_stream_pool_bytes = 0;
            d.cb_eval_user_data = &draft_trace;
            std::unique_ptr<llama_context, decltype(&llama_free)> draft(llama_init_from_model(model, d), &llama_free);
            if (!draft) throw std::runtime_error("cannot construct exact MTP draft");
            common_params_speculative params;
            params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            params.draft.ctx_tgt = target.get(); params.draft.ctx_dft = draft.get();
            params.draft.n_max = int32_t(length); params.draft.n_min = 0;
            params.draft.backend_sampling = false;
            common_speculative_ptr spec(common_speculative_init(params, 1));
            if (!spec) throw std::runtime_error("cannot construct production MTP driver");
            auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(target.get()));
            auto * draft_kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(draft.get()));
            if (!hybrid || !draft_kv) throw std::runtime_error("unsupported exact MTP memory architecture");
            auto * target_kv = hybrid->get_mem_attn();
            auto * stream = target_kv->get_kv_stream();
            if (bool(stream) != streaming) throw std::runtime_error("incorrect exact test route");
            const auto check = [&] { target_trace.check(); draft_trace.check(); };
            const auto phase = [&](const std::string & name) { target_trace.phase = draft_trace.phase = name; };
            const auto drain = [&] {
                llama_synchronize(target.get()); llama_synchronize(draft.get());
                if (stream && (!stream->complete() || !draft_kv->complete_mtp_publication()))
                    throw std::runtime_error("cannot drain exact KV publication");
                check();
            };
            const auto carry = [&](const std::string & name) {
                std::vector<uint8_t> bytes;
                if (!common_speculative_get_state(spec.get(), 0, bytes)) throw std::runtime_error("cannot read MTP carry state");
                oracle.bytes(name + "/carry", bytes.data(), bytes.size());
            };
            const auto recurrent = [&](const std::string & name) {
                drain();
                kv_stream_exact_state state;
                hybrid->get_mem_recr()->state_write(state, 0, 0);
                state.compare_with(oracle, name + "/recurrent");
            };
            const auto kv = [&](const std::string & name, size_t first, size_t count) {
                drain();
                kv_stream_exact_kv(oracle, name + "/target", target_kv, first, count);
                kv_stream_exact_kv(oracle, name + "/draft", draft_kv, first, count);
            };
            const auto submit = [&](const std::string & name, const llama_tokens & tokens, size_t first, bool decode) {
                phase(name);
                common_batch batch(target.get());
                for (size_t i = 0; i < tokens.size(); ++i) batch.add(tokens[i], llama_pos(first+i), 0, decode || i+1 == tokens.size());
                llama_set_kv_stream_decode(target.get(), decode);
                const size_t before = target_trace.outputs;
                const auto status = llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
                check();
                if (status) throw std::runtime_error("exact target execution failed: " + name);
                if (target_trace.outputs != before+1) throw std::runtime_error("missing exact target logits: " + name);
                if (!common_speculative_process(spec.get(), batch)) { check(); throw std::runtime_error("exact MTP catch-up failed: " + name); }
                drain();
            };
            for (size_t first = 0; first < prompt.size(); first += cfg.batch_rows) {
                const size_t count = std::min(cfg.batch_rows, prompt.size()-first);
                submit("prefill/" + std::to_string(first), {prompt.begin()+first, prompt.begin()+first+count}, first, false);
            }
            common_speculative_begin(spec.get(), 0, history);
            carry("prefill"); kv("prefill", 0, prompt.size()); recurrent("prefill");
            size_t position = prompt.size();
            auto * logits = llama_get_logits_ith(target.get(), -1);
            if (!logits) throw std::runtime_error("missing initial target logits");
            llama_token next = llama_token(std::max_element(logits, logits+vocabulary)-logits);
            bool target_streamed = false, draft_streamed = false;
            if (streaming) draft_trace.prediction_attention = [&] {
                ggml_kv_stream_span_plan_view view;
                if (!stream->has_mtp_layer() || !ggml_kv_stream_span_plan_get_view(stream->mtp_layer_plan(1), view))
                    throw std::runtime_error("prediction attention has no live MTP span lease");
                llama_kv_stream_runtime_diagnostics status;
                ggml_kv_stream_layout ring;
                if (!stream->runtime_diagnostics(status) || view.shape.page_tokens <= 0 ||
                        status.ring_slots > SIZE_MAX/size_t(view.shape.page_tokens) ||
                        ggml_kv_stream_layout_make(view.shape,size_t(status.ring_slots)*size_t(view.shape.page_tokens),ring).status !=
                            ggml_kv_stream_status::success)
                    throw std::runtime_error("invalid live MTP ring geometry");
                for (size_t i=0;i<view.count;++i)
                    draft_streamed |= view.spans[i].k_offset < ring.k_bytes && draft_kv->mtp_span_attention_calls() > 0;
            };
            for (size_t round = 0; round < cfg.rounds; ++round) {
                const auto name = "round/" + std::to_string(round);
                common_prompt_checkpoint checkpoint;
                checkpoint.update_tgt(target.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                std::vector<uint8_t> initial_carry;
                if (!common_speculative_get_state(spec.get(), 0, initial_carry)) throw std::runtime_error("missing initial MTP carry");
                llama_tokens predicted;
                common_speculative_get_draft_params(spec.get(), 0) = {
                    true, int32_t(length), llama_pos(position), next, &history, &predicted, nullptr, 0.0f, 123};
                phase(name + "/predict");
                const size_t before = draft_trace.outputs;
                common_speculative_draft(spec.get()); check(); drain();
                if (predicted.size() != length || draft_trace.outputs-before != length)
                    throw std::runtime_error("incomplete strict draft prediction");
                oracle.bytes(name + "/prediction_ids", predicted.data(), predicted.size()*sizeof(predicted[0]));
                kv_stream_exact_kv(oracle, name + "/predicted_KV", draft_kv, position, length);
                if (streaming) {
                    if (stream->has_mtp_layer())
                        throw std::runtime_error("prediction did not retire its MTP span lease after the final reader");
                }
                if (!llama_memory_seq_rm(llama_get_memory(draft.get()), 0, llama_pos(position), -1))
                    throw std::runtime_error("cannot discard predicted draft KV before catch-up");
                llama_tokens verified{next}; verified.insert(verified.end(), predicted.begin(), predicted.end());
                submit(name + "/verify", verified, position, true);
                carry(name + "/verify"); kv(name + "/verify", position, verified.size()); recurrent(name + "/verify");
                if (streaming) {
                    llama_kv_stream_runtime_diagnostics status;
                    if (!stream->runtime_diagnostics(status)) throw std::runtime_error("missing streaming diagnostics");
                    target_streamed |= status.streaming_active && status.active_pages > status.resident_pages_per_layer;
                }
                const size_t accepted = round%(length+1);
                const auto * selected = llama_get_logits_ith(target.get(), int32_t(accepted));
                if (!selected) throw std::runtime_error("missing verification logits");
                next = llama_token(std::max_element(selected, selected+vocabulary)-selected);
                const std::array<uint64_t, 2> decision{accepted, uint64_t(next)};
                oracle.bytes(name + "/decision", decision.data(), sizeof(decision));
                common_speculative_accept(spec.get(), 0, uint16_t(accepted));
                const size_t kept = accepted+1;
                if (!llama_memory_seq_rm(llama_get_memory(target.get()), 0, llama_pos(position+kept), -1) ||
                        !llama_memory_seq_rm(llama_get_memory(draft.get()), 0, llama_pos(position+kept), -1))
                    throw std::runtime_error("cannot apply strict acceptance truncation");
                carry(name + "/truncate"); kv(name + "/truncate", position, kept); recurrent(name + "/truncate");
                if (accepted < length) {
                    if (!llama_kv_stream_mtp_release(target.get()) ||
                            !checkpoint.load_tgt(target.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                            !llama_memory_seq_rm(llama_get_memory(target.get()), 0, llama_pos(position), -1) ||
                            !llama_memory_seq_rm(llama_get_memory(draft.get()), 0, llama_pos(position), -1))
                        throw std::runtime_error("cannot restore strict rejection checkpoint");
                    common_speculative_set_state(spec.get(), 0, initial_carry);
                    verified.resize(kept);
                    submit(name + "/replay", verified, position, true);
                    common_speculative_accept(spec.get(), 0, uint16_t(accepted));
                    carry(name + "/replay"); kv(name + "/replay", position, kept); recurrent(name + "/replay");
                    selected = llama_get_logits_ith(target.get(), int32_t(accepted));
                    if (!selected) throw std::runtime_error("missing replay logits");
                    next = llama_token(std::max_element(selected, selected+vocabulary)-selected);
                    oracle.bytes(name + "/replay_next", &next, sizeof(next));
                }
                history.insert(history.end(), verified.begin(), verified.begin()+kept);
                position += kept;
            }
            kv("final", 0, position); recurrent("final"); carry("final");
            if (streaming) {
                if (!target_streamed) {
                    llama_kv_stream_runtime_diagnostics status;
                    stream->runtime_diagnostics(status);
                    throw std::runtime_error("strict test never forced target streaming; pool=" +
                        std::to_string(status.pool_bytes) + " resident=" + std::to_string(status.resident_pages_per_layer) +
                        " active=" + std::to_string(status.active_pages) + " target=" + std::to_string(target_streamed) +
                        " draft=" + std::to_string(draft_streamed) + "; grow --exact-prefill or lower --exact-pool-mib");
                }
                // Concentrated placement can keep the auxiliary cache resident for an individual mode.
                observed_mtp_streaming |= draft_streamed;
                oracle.finish();
            }
            if (!llama_kv_stream_mtp_release(target.get())) throw std::runtime_error("cannot release final strict MTP lease");
            phase("");
            if (!streaming && !cfg.control_export.empty()) {
                std::unique_ptr<FILE,kv_stream_exact_file_free> output(std::fopen(cfg.control_export.string().c_str(),"wb"));
                if (!output) throw std::runtime_error("cannot create native MTP control");
                oracle.save_control(output.get(),profile);
            }
            t.out << (streaming ? "streamed" : "control") << " MTP=" << length << " prefix=" << cfg.prefill
                  << " rounds=" << cfg.rounds << " records=" << oracle.records() << " spool MiB="
                  << double(oracle.byte_count())/1048576 << " target_ring=" << target_streamed
                  << " mtp_ring=" << draft_streamed << '\n';
        }
        t.assert_true(oracle.records() >= cfg.rounds*4);
    });
    }
    t.assert_true(observed_mtp_streaming);
}

static void test_kv_stream_bit_exact_comparator(testing & t) {
    t.test("bit_exact_comparator_rejects_hidden_numerical_differences", [&](testing & t) {
        const std::array<uint32_t, 3> reference{0x3f800000, 0, 0x40000000};
        auto changed = reference;
        t.assert_true(kv_stream_bytes_equal(reference.data(), sizeof(reference), reference.data(), sizeof(reference)));
        changed[0]++;
        t.assert_true(!kv_stream_bytes_equal(reference.data(), sizeof(reference), changed.data(), sizeof(changed)));
        changed = reference;
        changed[1] = 0x80000000;
        t.assert_true(!kv_stream_bytes_equal(reference.data(), sizeof(reference), changed.data(), sizeof(changed)));
        changed = reference;
        changed[2] ^= 1;
        t.assert_true(!kv_stream_bytes_equal(reference.data(), sizeof(reference), changed.data(), sizeof(changed)));
        t.assert_true(!kv_stream_bytes_equal(reference.data(), sizeof(reference), reference.data(), sizeof(reference)-1));
        t.assert_true(!kv_stream_bytes_equal(reference.data(), sizeof(reference), nullptr, sizeof(reference)));
        t.assert_true(kv_stream_bytes_equal(nullptr, 0, nullptr, 0));
    });
    t.test("bit_exact_oracle_checks_framing_and_all_chunk_bytes", [&](testing & t) {
        const auto rejects = [](const std::function<void()> & operation) {
            try { operation(); } catch (const std::runtime_error &) { return true; }
            return false;
        };
        std::vector<uint8_t> reference(kv_stream_bit_oracle::chunk_bytes + 17, 0xa5);
        for (int mode = 0; mode < 8; ++mode) {
            kv_stream_bit_oracle oracle;
            oracle.bytes("KV", reference.data(), reference.size());
            oracle.bytes("empty", nullptr, 0);
            oracle.compare();
            auto changed = reference;
            if (mode == 0) {
                oracle.bytes("KV", reference.data(), reference.size());
                oracle.bytes("empty", nullptr, 0);
                oracle.finish();
                t.assert_equal(size_t(2), oracle.records());
                t.assert_equal(reference.size(), oracle.byte_count());
            } else if (mode == 1) {
                changed[kv_stream_bit_oracle::chunk_bytes] ^= 1;
                t.assert_true(rejects([&] { oracle.bytes("KV", changed.data(), changed.size()); }));
            } else if (mode == 2) {
                t.assert_true(rejects([&] { oracle.bytes("wrong", reference.data(), reference.size()); }));
            } else if (mode == 3) {
                t.assert_true(rejects([&] { oracle.bytes("KV", reference.data(), reference.size()-1); }));
            } else if (mode == 4) {
                t.assert_true(rejects([&] { oracle.finish(); }));
            } else if (mode == 5) {
                oracle.bytes("KV", reference.data(), reference.size());
                oracle.bytes("empty", nullptr, 0);
                t.assert_true(rejects([&] { oracle.bytes("extra", nullptr, 0); }));
            } else {
                changed[mode == 6 ? 0 : changed.size()-1] ^= 1;
                t.assert_true(rejects([&] { oracle.bytes("KV", changed.data(), changed.size()); }));
            }
        }
    });
    t.test("bit_exact_state_ignores_serialization_chunk_boundaries", [&](testing & t) {
        const std::array<uint8_t, 9> source{1, 2, 3, 4, 5, 6, 7, 8, 9};
        kv_stream_exact_state whole, split;
        whole.write(source.data(), source.size());
        split.write(source.data(), 3);
        split.write(source.data()+3, source.size()-3);
        kv_stream_bit_oracle oracle;
        whole.compare_with(oracle, "recurrent");
        oracle.compare();
        split.compare_with(oracle, "recurrent");
        oracle.finish();
        t.assert_equal(source.size(), whole.n_bytes());
        t.assert_equal(source.size(), split.n_bytes());
        bool rejected = false;
        try { split.write(nullptr, 1); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
    });
    t.test("default_exact_pool_can_hold_the_complete_auxiliary_history", [&](testing & t) {
        const kv_stream_exact_config config;
        const size_t capacity = (config.prefill+config.rounds*4+255)/256*256;
        ggml_kv_stream_layout layer;
        t.assert_true(ggml_kv_stream_layout_make({config.key,config.value,256,256,4,256,128},capacity,layer).status ==
            ggml_kv_stream_status::success);
        t.assert_true(config.pool_mib*1048576 >= layer.bytes);
    });
    t.test("MTP_control_files_require_one_explicit_profile_and_one_direction", [&](testing & t) {
        const auto parse=[](std::vector<std::string> fields) {
            std::vector<char *> args;
            for (auto & field:fields) args.push_back(field.data());
            return kv_stream_exact_arguments(int(args.size()),args.data());
        };
        const auto out=parse({"test","--model","model","--mtp-bit-exact","--exact-length","3",
            "--exact-control-export","native.bin"});
        t.assert_equal(std::string("native.bin"),out.control_export.string());
        const auto in=parse({"test","--model","model","--mtp-bit-exact","--exact-length","3",
            "--exact-control-import","native.bin"});
        t.assert_equal(std::string("native.bin"),in.control_import.string());
        for (const auto & tail:std::vector<std::vector<std::string>>{
                {"--exact-control-export","native.bin"},
                {"--exact-length","3","--exact-control-export",""},
                {"--exact-length","3","--exact-control-export","native.bin","--exact-control-import","native.bin"},
                {"--exact-length","3","--exact-control-import"}}) {
            std::vector<std::string> fields{"test","--model","model","--mtp-bit-exact"};
            fields.insert(fields.end(),tail.begin(),tail.end());
            bool rejected=false;
            try { parse(fields); } catch (const std::exception &) { rejected=true; }
            t.assert_true(rejected);
        }
    });
    t.test("bit_exact_arguments_accept_explicit_cache_formats", [&](testing & t) {
        std::vector<std::string> fields{"test", "--model", "model", "--mtp-bit-exact",
            "--test-cache-k", "q5_1", "--test-cache-v", "q8_0"};
        std::vector<char *> arguments;
        for (auto & field : fields) arguments.push_back(field.data());
        const auto config = kv_stream_exact_arguments(int(arguments.size()), arguments.data());
        t.assert_equal(size_t(16381), config.prefill);
        t.assert_equal(size_t(256), config.batch_rows);
        t.assert_equal(size_t(0), config.length);
        t.assert_true(config.key == GGML_TYPE_Q5_1 && config.value == GGML_TYPE_Q8_0);
        for (auto key : {"f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"})
            for (auto value : {"f32", "f16", "bf16", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"}) {
                fields[5] = key; fields[7] = value;
                arguments.clear();
                for (auto & field : fields) arguments.push_back(field.data());
                const auto pair = kv_stream_exact_arguments(int(arguments.size()), arguments.data());
                t.assert_equal(std::string(key), std::string(ggml_type_name(pair.key)));
                t.assert_equal(std::string(value), std::string(ggml_type_name(pair.value)));
            }
    });
    t.test("exact_microbatch_profile_can_isolate_ring_from_reclaimed_prefill_memory", [&](testing & t) {
        std::vector<std::string> fields{"test", "--model", "model", "--mtp-bit-exact",
            "--exact-batch-size", "8", "--exact-length", "3"};
        std::vector<char *> arguments;
        for (auto & field : fields) arguments.push_back(field.data());
        const auto config = kv_stream_exact_arguments(int(arguments.size()), arguments.data());
        t.assert_equal(size_t(16381), config.prefill);
        t.assert_equal(size_t(8), config.batch_rows);
        t.assert_equal(size_t(3), config.length);
    });
    t.test("bit_exact_arguments_reject_unbounded_or_incomplete_profiles", [&](testing & t) {
        const auto rejects = [](std::vector<std::string> fields) {
            std::vector<char *> argv;
            for (auto & field : fields) argv.push_back(field.data());
            try { kv_stream_exact_arguments(int(argv.size()), argv.data()); }
            catch (const std::exception &) { return true; }
            return false;
        };
        for (const char * value : {"0", "-1", "32769", "18446744073709551616", "16381junk"})
            t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--exact-prefill", value}));
        t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--exact-rounds", "3"}));
        t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--exact-pool-mib", "1025"}));
        for (const char * value : {"0", "3", "4", "257"})
            t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--exact-batch-size", value}));
        for (const char * value : {"0", "4"})
            t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--exact-length", value}));
        t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--exact-prefill"}));
        t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--unknown"}));
        for (const char * value : {"q8_1", "q2_k", "i32", "", "unknown"})
            t.assert_true(rejects({"test", "--model", "model", "--mtp-bit-exact", "--test-cache-k", value}));
    });
    t.test("bit_exact_corpus_is_bounded_before_allocation", [&](testing & t) {
        std::istringstream valid("article");
        t.assert_equal(std::string("article"), kv_stream_exact_text(valid));
        std::istringstream empty;
        bool rejected = false;
        try { kv_stream_exact_text(empty); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
        struct oversized : std::streambuf {
            pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
                return pos_type(std::streamoff(64*1048576+1));
            }
            int_type underflow() override { throw std::logic_error("oversized corpus was read"); }
        } buffer;
        std::istream huge(&buffer);
        rejected = false;
        try { kv_stream_exact_text(huge); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
        std::istream nonseekable(nullptr);
        rejected = false;
        try { kv_stream_exact_text(nonseekable); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true(rejected);
    });
}
