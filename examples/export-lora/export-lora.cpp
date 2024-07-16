
#include "common.h"
#include "ggml.h"
#include "ggml-alloc.h"

#include <vector>
#include <string>
#include <thread>

struct lora_info {
    std::string filename;
    float scale;
};

struct export_lora_params {
    std::string fn_model_base;
    std::string fn_model_out;
    std::vector<struct lora_info> lora;
    int n_threads;
};

struct lora_data {
    struct lora_info     info;
    std::vector<uint8_t> data;
    struct ggml_context * ctx;

    uint32_t lora_r;
    uint32_t lora_alpha;
};

struct llama_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE * fp;
    size_t size;

    llama_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            size = 0;
        } else {
            seek(0, SEEK_END);
            size = tell();
            seek(0, SEEK_SET);
        }
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64) offset, whence);
#else
        int ret = std::fseek(fp, (long) offset, whence);
#endif
        GGML_ASSERT(ret == 0); // same
    }

    void read_raw(void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, size, 1, fp);
        if (ferror(fp)) {
            die_fmt("read error: %s", strerror(errno));
        }
        if (ret != 1) {
            die("unexpectedly reached end of file");
        }
    }

    std::uint32_t read_u32() {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    void write_raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, size, 1, fp);
        if (ret != 1) {
            die_fmt("write error: %s", strerror(errno));
        }
    }

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    bool eof() {
        return tell() >= size;
    }

    ~llama_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};

static struct export_lora_params get_default_export_lora_params() {
    struct export_lora_params result;
    result.fn_model_base = "";
    result.fn_model_out  = "";
    result.n_threads = GGML_DEFAULT_N_THREADS;
    return result;
}

static void export_lora_print_usage(int /*argc*/, char ** argv, const struct export_lora_params * params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                         show this help message and exit\n");
    fprintf(stderr, "  -m FNAME, --model-base FNAME       model path from which to load base model (default '%s')\n", params->fn_model_base.c_str());
    fprintf(stderr, "  -o FNAME, --model-out FNAME        path to save exported model (default '%s')\n", params->fn_model_out.c_str());
    fprintf(stderr, "  -l FNAME, --lora FNAME             apply LoRA adapter\n");
    fprintf(stderr, "  -s FNAME S, --lora-scaled FNAME S  apply LoRA adapter with user defined scaling S\n");
    fprintf(stderr, "  -t N, --threads N                  number of threads to use during computation (default: %d)\n", params->n_threads);
}

static bool export_lora_params_parse(int argc, char ** argv, struct export_lora_params * params) {
    bool invalid_param = false;
    std::string arg;
    struct export_lora_params default_params = get_default_export_lora_params();
    const std::string arg_prefix = "--";

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-m" || arg == "--model-base") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_base = argv[i];
        } else if (arg == "-o" || arg == "--model-out") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->fn_model_out = argv[i];
        } else if (arg == "-l" || arg == "--lora") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            struct lora_info lora;
            lora.filename = argv[i];
            lora.scale = 1.0f;
            params->lora.push_back(lora);
        } else if (arg == "-s" || arg == "--lora-scaled") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            struct lora_info lora;
            lora.filename = argv[i];
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            lora.scale = std::stof(argv[i]);
            params->lora.push_back(lora);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params->n_threads = std::stoi(argv[i]);
            if (params->n_threads <= 0) {
                params->n_threads = std::thread::hardware_concurrency();
            }
        } else if (arg == "-h" || arg == "--help") {
            export_lora_print_usage(argc, argv, &default_params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: '%s'\n", arg.c_str());
            export_lora_print_usage(argc, argv, &default_params);
            exit(1);
        }
    }

    if (params->fn_model_base == default_params.fn_model_base) {
        fprintf(stderr, "error: please specify a filename for model-base.\n");
        export_lora_print_usage(argc, argv, &default_params);
        exit(1);
    }
    if (params->fn_model_out == default_params.fn_model_out) {
        fprintf(stderr, "error: please specify a filename for model-out.\n");
        export_lora_print_usage(argc, argv, &default_params);
        exit(1);
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: '%s'\n", arg.c_str());
        export_lora_print_usage(argc, argv, &default_params);
        exit(1);
    }
    return true;
}

static void free_lora(struct lora_data * lora) {
    if (lora->ctx != NULL) {
        ggml_free(lora->ctx);
    }
    delete lora;
}

static struct lora_data * load_lora(struct lora_info * info) {
    struct lora_data * result = new struct lora_data;
    result->info = *info;
    result->ctx = NULL;
    result->lora_r     = 1;
    result->lora_alpha = 1;

    struct llama_file file(info->filename.c_str(), "rb");
    if (file.fp == NULL) {
        fprintf(stderr, "warning: Could not open lora adapter '%s'. Ignoring this adapter.\n",
            info->filename.c_str());
        free_lora(result);
        return NULL;
    }

    struct ggml_init_params params_ggml;
    params_ggml.mem_size   = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE;
    params_ggml.mem_buffer = NULL;
    params_ggml.no_alloc   = true;
    result->ctx = ggml_init(params_ggml);

    uint32_t magic   = file.read_u32();
    if (magic != LLAMA_FILE_MAGIC_GGLA) {
        die_fmt("unexpected lora header file magic in '%s'", info->filename.c_str());
    }
    uint32_t version = file.read_u32();
    if (version != 1) {
        die_fmt("unexpected lora file version '%u' in '%s'", (unsigned) version, info->filename.c_str());
    }
    result->lora_r     = file.read_u32();
    result->lora_alpha = file.read_u32();
    // read tensor infos from file
    std::vector<char> name_buf;
    std::vector<struct ggml_tensor *> tensors;
    std::vector<size_t> tensors_offset;
    size_t total_nbytes_pad = 0;
    while(!file.eof()) {
        int64_t ne[4]   = {1,1,1,1};
        uint32_t n_dims  = file.read_u32();
        uint32_t namelen = file.read_u32();
        uint32_t type    = file.read_u32();
        for (uint32_t k = 0; k < n_dims; ++k) {
            ne[k] = (int64_t)file.read_u32();
        }
        name_buf.clear();
        name_buf.resize(namelen + 1, '\0');
        file.read_raw(name_buf.data(), namelen);
        file.seek((0-file.tell()) & 31, SEEK_CUR);
        size_t offset = file.tell();
        struct ggml_tensor * tensor = ggml_new_tensor(result->ctx, (enum ggml_type) type, n_dims, ne);
        ggml_set_name(tensor, name_buf.data());
        size_t nbytes     = ggml_nbytes(tensor);
        size_t nbytes_pad = ggml_nbytes_pad(tensor);
        total_nbytes_pad += nbytes_pad;
        tensors.push_back(tensor);
        tensors_offset.push_back(offset);
        file.seek(nbytes, SEEK_CUR);
    }
    // read tensor data
    result->data.resize(total_nbytes_pad);
    size_t data_offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        struct ggml_tensor * tensor = tensors[i];
        size_t offset     = tensors_offset[i];
        size_t nbytes     = ggml_nbytes(tensor);
        size_t nbytes_pad = ggml_nbytes_pad(tensor);
        file.seek(offset, SEEK_SET);
        tensor->data = result->data.data() + data_offset;
        file.read_raw(tensor->data, nbytes);
        data_offset += nbytes_pad;
    }
    return result;
}


static struct ggml_cgraph * build_graph_lora(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    struct ggml_tensor * lora_a,
    struct ggml_tensor * lora_b,
    float scaling
) {
    struct ggml_tensor * ab = ggml_mul_mat(ctx, lora_a, lora_b);
    if (scaling != 1.0f) {
        ab = ggml_scale(ctx, ab, scaling);
    }
    struct ggml_tensor * res = ggml_add_inplace(ctx, tensor, ab);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand (gf, res);
    return gf;
}

static bool apply_lora(struct ggml_tensor * tensor, struct lora_data * lora, int n_threads) {
    if (lora->ctx == NULL) {
        return false;
    }
    std::string name = ggml_get_name(tensor);
    std::string name_a = name + std::string(".loraA");
    std::string name_b = name + std::string(".loraB");
    struct ggml_tensor * lora_a = ggml_get_tensor(lora->ctx, name_a.c_str());
    struct ggml_tensor * lora_b = ggml_get_tensor(lora->ctx, name_b.c_str());
    if (lora_a == NULL || lora_b == NULL) {
        return false;
    }

    float scaling = lora->info.scale * (float)lora->lora_alpha / (float)lora->lora_r;

    struct ggml_init_params params;
    params.mem_size   = GGML_OBJECT_SIZE + ggml_graph_overhead() + ggml_tensor_overhead()*4 + GGML_MEM_ALIGN*5;
    params.mem_buffer = NULL;
    params.no_alloc   = true;
    struct ggml_context * ctx = NULL;
    struct ggml_gallocr * alloc = NULL;
    struct ggml_cgraph  * gf = NULL;

    ctx   = ggml_init(params);
    alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    gf    = build_graph_lora(ctx, tensor, lora_a, lora_b, scaling);

    ggml_gallocr_alloc_graph(alloc, gf);

    struct ggml_cplan cplan = ggml_graph_plan(gf, n_threads);
    static std::vector<uint8_t> data_work;
    data_work.resize(cplan.work_size);
    cplan.work_data = data_work.data();

    ggml_graph_compute(gf, &cplan);

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return true;
}

static void export_lora(struct export_lora_params * params) {
    // load all loras
    std::vector<struct lora_data *> loras;
    for (size_t i = 0; i < params->lora.size(); ++i) {
        struct lora_data * lora = load_lora(&params->lora[i]);
        if (lora != NULL) {
            loras.push_back(lora);
        }
    }
    if (loras.size() == 0) {
        fprintf(stderr, "warning: no lora adapters will be applied.\n");
    }

    // open input file
    struct llama_file fin(params->fn_model_base.c_str(), "rb");
    if (!fin.fp) {
        die_fmt("Could not open file '%s'\n", params->fn_model_base.c_str());
    }

    // open base model gguf, read tensors without their data
    struct ggml_context * ctx_in;
    struct gguf_init_params params_gguf;
    params_gguf.no_alloc = true;
    params_gguf.ctx      = &ctx_in;
    struct gguf_context * gguf_in = gguf_init_from_file(params->fn_model_base.c_str(), params_gguf);

    // create new gguf
    struct gguf_context * gguf_out = gguf_init_empty();

    // copy meta data from base model: kv and tensors
    gguf_set_kv(gguf_out, gguf_in);
    int n_tensors = gguf_get_n_tensors(gguf_in);
    for (int i=0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_in, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_in, name);
        gguf_add_tensor(gguf_out, tensor);
    }

    // create output file
    struct llama_file fout(params->fn_model_out.c_str(), "wb");
    if (!fout.fp) {
        die_fmt("Could not create file '%s'\n", params->fn_model_out.c_str());
    }

    // write gguf meta data
    std::vector<uint8_t> meta;
    meta.resize(gguf_get_meta_size(gguf_out));
    gguf_get_meta_data(gguf_out, meta.data());
    fout.write_raw(meta.data(), meta.size());

    std::vector<uint8_t> data;
    std::vector<uint8_t> padding;
    for (int i=0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_in, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_in, name);

        // read tensor data
        data.resize(ggml_nbytes(tensor));
        tensor->data = data.data();
        size_t offset = gguf_get_tensor_offset(gguf_in, i);
        fin.seek(offset + meta.size(), SEEK_SET);
        fin.read_raw(data.data(), data.size());

        // apply all loras
        for (size_t k = 0; k < loras.size(); ++k) {
            apply_lora(tensor, loras[k], params->n_threads);
        }

        // write tensor data + padding
        padding.clear();
        padding.resize(GGML_PAD(data.size(), gguf_get_alignment(gguf_out)) - data.size(), 0);

        GGML_ASSERT(fout.tell() == offset + meta.size());
        // fout.seek(offset + meta.size(), SEEK_SET);
        fout.write_raw(data.data(), data.size());
        fout.write_raw(padding.data(), padding.size());

        if (i % 2 == 0) {
            printf(".");
        }
    }
    printf("\n");

    // close gguf
    gguf_free(gguf_out);
    gguf_free(gguf_in);

    // free loras
    for (size_t i = 0; i < loras.size(); ++i) {
        free_lora(loras[i]);
    }
}

int main(int argc, char ** argv) {
    struct export_lora_params params = get_default_export_lora_params();

    if (!export_lora_params_parse(argc, argv, &params)) {
        return 1;
    }

    export_lora(&params);

    return 0;
}
