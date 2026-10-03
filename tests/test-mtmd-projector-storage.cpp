#include "../tools/mtmd/mtmd-projector-storage.h"
#include "../tools/mtmd/mtmd-helper.h"
#include "../tools/mtmd/mtmd-embeddings.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "testing.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <fstream>
#include <thread>
#include <atomic>

struct weight_buffer_probe {
    inline static weight_buffer_probe * active = nullptr;
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::free_buffer) original;
    size_t frees = 0;
    explicit weight_buffer_probe(ggml_backend_buffer_t buffer) : buffer(buffer),original(buffer->iface.free_buffer) {
        GGML_ASSERT(!active); active = this;
        buffer->iface.free_buffer = [](ggml_backend_buffer_t buffer) {
            ++active->frees;
            if (active->original) active->original(buffer);
        };
    }
    ~weight_buffer_probe() { if (!frees) buffer->iface.free_buffer = original; active = nullptr; }
};

struct weight_allocation_fault {
    ggml_backend_buffer_type_t type = ggml_backend_cpu_buffer_type();
    decltype(ggml_backend_buffer_type_i::alloc_buffer) original = type->iface.alloc_buffer;
    weight_allocation_fault() { type->iface.alloc_buffer = [](ggml_backend_buffer_type_t,size_t) -> ggml_backend_buffer_t { return nullptr; }; }
    ~weight_allocation_fault() { type->iface.alloc_buffer = original; }
};

struct source_fixture {
    ggml_context_ptr tensors{ggml_init({65536,nullptr,false})};
    gguf_context_ptr gguf{gguf_init_empty()};
    ggml_tensor * a = ggml_new_tensor_1d(tensors.get(),GGML_TYPE_F32,32);
    ggml_tensor * b = ggml_new_tensor_1d(tensors.get(),GGML_TYPE_F16,32);
    size_t data_offset = 0;
    source_fixture() {
        ggml_set_name(a,"v.first"); ggml_set_name(b,"a.second");
        std::memset(a->data,0x3c,ggml_nbytes(a)); std::memset(b->data,0x5a,ggml_nbytes(b));
        gguf_add_tensor(gguf.get(),a); gguf_add_tensor(gguf.get(),b);
    }
    std::shared_ptr<const mtmd_projector_source> source(bool truncated = false) {
        FILE * file = std::tmpfile(); GGML_ASSERT(file);
        GGML_ASSERT(gguf_write_to_file_ptr(gguf.get(),file,truncated));
        std::rewind(file);
        gguf_context_ptr parsed(gguf_init_from_file_ptr(file,{true,nullptr})); GGML_ASSERT(parsed);
        data_offset = gguf_get_data_offset(parsed.get());
        return mtmd_projector_source::from_file(file,parsed.get());
    }
    ggml_context_ptr context(bool both = true) {
        ggml_context_ptr result(ggml_init({65536,nullptr,true}));
        ggml_set_name(ggml_dup_tensor(result.get(),a),a->name);
        if (both) ggml_set_name(ggml_dup_tensor(result.get(),b),b->name);
        return result;
    }
};

int main(int argc,char ** argv) {
    const char * model_path = nullptr, * projector_path = nullptr, * baseline = nullptr, * save = nullptr;
    bool cuda = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i],"--cuda")) cuda = true;
        else if (!std::strcmp(argv[i],"--model") && i+1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i],"--mmproj") && i+1 < argc) projector_path = argv[++i];
        else if (!std::strcmp(argv[i],"--baseline") && i+1 < argc) baseline = argv[++i];
        else if (!std::strcmp(argv[i],"--save-baseline") && i+1 < argc) save = argv[++i];
        else return 2;
    }
    testing t;
    t.test("source_manifest_and_bytes_outlive_the_loader_metadata", [](testing & t) {
        std::shared_ptr<const mtmd_projector_source> source;
        {
            source_fixture f; source = f.source();
            if (!t.assert_true(bool(source))) return;
            const auto * a = source->find("v.first");
            if (!t.assert_true(a != nullptr)) return;
            t.assert_equal(size_t(128),a->bytes);
            t.assert_true(a->type == GGML_TYPE_F32 && a->shape[0] == 32);
            t.assert_equal(f.data_offset + gguf_get_tensor_offset(f.gguf.get(),0),a->offset);
        }
        std::vector<uint8_t> output(128,0);
        t.assert_true(source->read("v.first",output.data(),output.size()));
        t.assert_true(std::all_of(output.begin(),output.end(),[](uint8_t b) { return b == 0x3c; }));
        t.assert_true(!source->read("missing",output.data(),output.size()));
        t.assert_true(!source->read("v.first",output.data(),127));
        t.assert_true(!source->read(nullptr,output.data(),128));
        t.assert_true(!source->read("v.first",nullptr,128));
        t.assert_true(!source->find(nullptr));
        t.assert_true(std::all_of(output.begin(),output.end(),[](uint8_t b) { return b == 0x3c; }));
    });
    t.test("metadata_rejects_wrong_shapes_names_types_and_bound_descriptors", [](testing & t) {
        source_fixture f; auto source = f.source();
        if (!t.assert_true(bool(source))) return;
        auto changed = f.context(); ggml_get_first_tensor(changed.get())->ne[0] = 16;
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_set_name(ggml_get_first_tensor(changed.get()),"missing");
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_get_first_tensor(changed.get())->type = GGML_TYPE_F16;
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_get_first_tensor(changed.get())->data = f.a->data;
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        changed = f.context(); ggml_set_name(ggml_get_next_tensor(changed.get(),ggml_get_first_tensor(changed.get())),"v.first");
        t.assert_true(!mtmd_projector_metadata::create(std::move(changed),source));
        t.assert_true(!mtmd_projector_metadata::create({},source));
        t.assert_true(!mtmd_projector_metadata::create(f.context(),{}));
        t.assert_true(bool(mtmd_projector_metadata::create(f.context(false),source)));
    });
    t.test("shared_weights_release_once_and_metadata_remains_unbound", [](testing & t) {
        source_fixture f;
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        if (!t.assert_true(bool(metadata))) return;
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        if (!t.assert_true(bool(resident))) return;
        auto * a = ggml_get_first_tensor(metadata->context());
        t.assert_true(a->buffer && a->data);
        std::vector<uint8_t> bytes(128); ggml_backend_tensor_get(a,bytes.data(),0,bytes.size());
        t.assert_true(std::all_of(bytes.begin(),bytes.end(),[](uint8_t b) { return b == 0x3c; }));
        t.assert_true(ggml_backend_buffer_get_usage(resident->buffer()) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto another = resident;
        weight_buffer_probe probe(resident->buffer());
        t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type()));
        std::weak_ptr<mtmd_projector_weights> weak = resident;
        resident.reset(); t.assert_true(!weak.expired() && a->data != nullptr);
        another.reset(); t.assert_true(weak.expired());
        t.assert_equal(size_t(1),probe.frees);
        for (auto * tensor = a; tensor; tensor = ggml_get_next_tensor(metadata->context(),tensor))
            t.assert_true(!tensor->buffer && !tensor->data && !tensor->extra);
        t.assert_true(metadata->source()->read("v.first",bytes.data(),bytes.size()));
    });
    t.test("failed_and_cancelled_loading_returns_candidate_bindings", [](testing & t) {
        source_fixture f;
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        if (!t.assert_true(bool(metadata))) return;
        t.assert_true(!mtmd_projector_weights::allocate(metadata,nullptr));
        {
            weight_allocation_fault fault;
            t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type()));
        }
        auto cancel = [](float,void *) { return false; };
        t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,cancel));
        size_t callbacks = 0;
        auto late = [](float,void * p) { return ++*static_cast<size_t *>(p) == 1; };
        t.assert_true(!mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,late,&callbacks));
        t.assert_equal(size_t(2),callbacks);
        auto throwing = [](float progress,void *) { if (progress > 0) throw std::runtime_error("cancel"); return true; };
        try { mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,throwing); t.assert_true(false); }
        catch (const std::runtime_error &) { t.assert_true(true); }
        for (auto * tensor = ggml_get_first_tensor(metadata->context()); tensor; tensor = ggml_get_next_tensor(metadata->context(),tensor))
            t.assert_true(!tensor->buffer && !tensor->data && !tensor->extra);
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type());
        t.assert_true(bool(resident));
        auto truncated = mtmd_projector_metadata::create(f.context(),f.source(true));
        t.assert_true(truncated && !mtmd_projector_weights::allocate(truncated,ggml_backend_cpu_buffer_type()));
        t.assert_true(!ggml_get_first_tensor(truncated->context())->data);
    });
    t.test("loading_callbacks_cannot_reenter_the_same_metadata", [](testing & t) {
        source_fixture f; auto metadata = mtmd_projector_metadata::create(f.context(),f.source());
        if (!t.assert_true(bool(metadata))) return;
        struct probe { std::shared_ptr<mtmd_projector_metadata> metadata; size_t calls = 0; bool admitted = false; } p{metadata};
        auto callback = [](float,void * data) {
            auto & p = *static_cast<probe *>(data); ++p.calls;
            p.admitted |= bool(mtmd_projector_weights::allocate(p.metadata,ggml_backend_cpu_buffer_type()));
            return true;
        };
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),false,callback,&p);
        t.assert_true(resident && !p.admitted && p.calls == 3);
    });
    t.test("temporary_metadata_and_skipped_upload_keep_lifetime_boundaries", [](testing & t) {
        source_fixture f;
        auto cancel = [](float p,void *) { return p == 0; };
        t.assert_true(!mtmd_projector_weights::allocate(mtmd_projector_metadata::create(f.context(),f.source()),
            ggml_backend_cpu_buffer_type(),false,cancel));
        auto metadata = mtmd_projector_metadata::create(f.context(),f.source(true));
        if (!t.assert_true(bool(metadata))) return;
        auto resident = mtmd_projector_weights::allocate(metadata,ggml_backend_cpu_buffer_type(),true);
        if (!t.assert_true(bool(resident))) return;
        auto weak = std::weak_ptr<mtmd_projector_metadata>(metadata);
        metadata.reset();
        t.assert_true(!weak.expired());
        resident.reset();
        t.assert_true(weak.expired());
    });
    t.test("shared_source_reads_do_not_race_the_file_cursor", [](testing & t) {
        source_fixture f; auto source = f.source();
        if (!t.assert_true(bool(source))) return;
        std::atomic<bool> ok{true};
        auto read = [&](const char * name,size_t bytes,uint8_t value) {
            std::vector<uint8_t> data(bytes);
            for (int i = 0; i < 64; ++i) {
                if (!source->read(name,data.data(),bytes) ||
                        !std::all_of(data.begin(),data.end(),[&](uint8_t b) { return b == value; })) ok = false;
            }
        };
        std::thread first(read,"v.first",128,0x3c),second(read,"a.second",64,0x5a);
        first.join(); second.join();
        t.assert_true(ok.load());
    });
    t.test("invalid_source_handles_and_missing_paths_are_rejected", [](testing & t) {
        source_fixture f;
        t.assert_true(!mtmd_projector_source::from_file(nullptr,f.gguf.get()));
        t.assert_true(!mtmd_projector_source::from_file(std::tmpfile(),nullptr));
        t.assert_true(!mtmd_projector_source::open(nullptr,f.gguf.get()));
        t.assert_true(!mtmd_projector_source::open("",f.gguf.get()));
        t.assert_true(!mtmd_projector_source::from_file(std::tmpfile(),f.gguf.get()));
    });
    if (model_path && projector_path) t.test("real_projector_encoding_matches_the_original_eager_loader", [&](testing & t) {
        ggml_backend_load_all();
        auto model_params = llama_model_default_params();
        model_params.no_alloc = true; model_params.n_gpu_layers = 0;
        model_params.load_mode = LLAMA_LOAD_MODE_NONE; model_params.use_extra_bufts = false;
        llama_model_ptr model(llama_model_load_from_file(model_path,model_params));
        if (!t.assert_true(bool(model))) return;
        auto params = mtmd_context_params_default();
        params.use_gpu = cuda; params.warmup = false; params.n_threads = 4;
        params.image_min_tokens = 64; params.image_max_tokens = 256;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        mtmd::context_ptr projector(mtmd_init_from_file(projector_path,model.get(),params));
        if (!t.assert_true(bool(projector))) return;
        std::vector<uint8_t> pixels(256*256*3);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = uint8_t((i*7+i/101)%256);
        mtmd::bitmap_ptr image(mtmd_bitmap_init(256,256,pixels.data()));
        const auto * bitmap = image.get();
        const std::string prompt = mtmd_get_marker(projector.get());
        mtmd_input_text input{prompt.c_str(),prompt.size(),true,true};
        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        if (!t.assert_equal(0,mtmd_tokenize(projector.get(),chunks.get(),&input,&bitmap,1))) return;
        mtmd::batch_ptr batch(mtmd_batch_init(projector.get()));
        const mtmd_input_chunk * chunk = nullptr;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) {
            auto * entry = mtmd_input_chunks_get(chunks.get(),i);
            if (mtmd_input_chunk_get_type(entry) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                chunk = entry;
                if (!t.assert_equal(0,mtmd_batch_add_chunk(batch.get(),entry))) return;
            }
        }
        if (!t.assert_true(chunk != nullptr) || !t.assert_equal(0,mtmd_batch_encode(batch.get()))) return;
        mtmd_embedding_view output;
        if (!t.assert_true(mtmd_batch_acquire_output_embd(batch.get(),chunk,output))) return;
        const size_t bytes = output.n_tokens()*output.n_embd()*sizeof(float);
        if (save) {
            std::ofstream file(save,std::ios::binary);
            file.write(reinterpret_cast<const char *>(output.data()),std::streamsize(bytes));
            t.assert_true(bool(file));
        }
        if (baseline) {
            std::vector<uint8_t> expected(bytes);
            std::ifstream file(baseline,std::ios::binary);
            file.read(reinterpret_cast<char *>(expected.data()),std::streamsize(bytes));
            if (!t.assert_true(bool(file) && file.peek() == EOF)) return;
            t.assert_true(std::memcmp(expected.data(),output.data(),bytes) == 0);
        }
        if (!save) {
            auto weights = mtmd_acquire_projector_weights(projector.get());
            if (!t.assert_true(bool(weights))) return;
            auto metadata = weights->metadata();
            auto * tensor = ggml_get_first_tensor(metadata->context());
            const size_t sample_bytes = std::min(size_t(64),ggml_nbytes(tensor));
            std::vector<uint8_t> expected(sample_bytes),actual(sample_bytes);
            ggml_backend_tensor_get(tensor,expected.data(),0,sample_bytes);
            batch.reset(); projector.reset();
            // The resident owner outlives its projector scheduler and backend handle.
            t.assert_true(tensor->data && tensor->buffer);
            ggml_backend_tensor_get(tensor,actual.data(),0,sample_bytes);
            t.assert_true(actual == expected);
            weights.reset();
            t.assert_true(!tensor->buffer && !tensor->data && !tensor->extra);
            std::vector<uint8_t> file_bytes(ggml_nbytes(tensor));
            t.assert_true(metadata->source()->read(tensor->name,file_bytes.data(),file_bytes.size()));
            t.assert_true(std::equal(expected.begin(),expected.end(),file_bytes.begin()));
            t.assert_true(output.data() != nullptr);
        }
    });
    return t.summary();
}
