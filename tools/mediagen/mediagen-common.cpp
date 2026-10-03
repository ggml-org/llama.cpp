#include "mediagen-impl.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>

using json = nlohmann::ordered_json;

//
// logging
//

void mediagen_logger::log(ggml_log_level level, const char * fmt, ...) {
    if (level < verbosity) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

std::string mg_format(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    std::vector<char> buf(size + 1);
    vsnprintf(buf.data(), size + 1, fmt, ap2);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

//
// mg_weights
//

ggml_tensor * mg_weights::get(const std::string & name, bool required) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        if (required) {
            throw std::runtime_error(mg_format("missing tensor '%s'", name.c_str()));
        }
        return nullptr;
    }
    return it->second;
}

bool mg_tensor_to_f32(const ggml_tensor * t, int64_t n, std::vector<float> & out) {
    if (t->ne[0] < n || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
        MG_ERR("%s: unexpected tensor '%s' (%s, %lld elements, need %lld)\n", __func__, t->name, ggml_type_name(t->type), (long long) t->ne[0], (long long) n);
        return false;
    }
    std::vector<uint8_t> buf(ggml_row_size(t->type, n));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), buf.data(), buf.size());
    } else {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) buf.data(), out.data(), n);
    }
    return true;
}

// fill d.nbytes from d.ne / d.type, false on invalid or overflowing dims
static bool mg_desc_finalize(mg_tensor_desc & d) {
    if (d.ne.empty() || d.ne.size() > 5 || d.type < 0 || d.type >= GGML_TYPE_COUNT || ggml_type_size(d.type) == 0) {
        return false;
    }
    const int64_t blck = ggml_blck_size(d.type);
    if (d.ne[0] <= 0 || d.ne[0] % blck != 0) {
        return false;
    }
    int64_t n = d.ne[0] / blck;
    for (size_t i = 1; i < d.ne.size(); i++) {
        if (d.ne[i] <= 0 || n > INT64_MAX / d.ne[i]) {
            return false;
        }
        n *= d.ne[i];
    }
    if (n > (int64_t) (SIZE_MAX / ggml_type_size(d.type))) {
        return false;
    }
    d.nbytes = (size_t) n * ggml_type_size(d.type);
    return true;
}

//
// safetensors
//

static ggml_type st_dtype_to_ggml(const std::string & dtype) {
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "F16")  return GGML_TYPE_F16;
    if (dtype == "F32")  return GGML_TYPE_F32;
    if (dtype == "F64")  return GGML_TYPE_F64;
    if (dtype == "I32")  return GGML_TYPE_I32;
    if (dtype == "I64")  return GGML_TYPE_I64;
    return GGML_TYPE_COUNT;
}

bool mg_safetensors_read_header(const std::string & path, std::vector<mg_tensor_desc> & out, std::string * metadata, size_t * data_start) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        MG_ERR("%s: failed to open '%s'\n", __func__, path.c_str());
        return false;
    }
    uint64_t n_header = 0;
    fin.read((char *) &n_header, sizeof(n_header));
    if (!fin || n_header == 0 || n_header > (1u << 30)) {
        MG_ERR("%s: invalid safetensors header in '%s'\n", __func__, path.c_str());
        return false;
    }
    std::string header(n_header, '\0');
    fin.read(header.data(), n_header);
    if (!fin) {
        MG_ERR("%s: truncated safetensors header in '%s'\n", __func__, path.c_str());
        return false;
    }
    json j;
    try {
        j = json::parse(header);
    } catch (const std::exception & e) {
        MG_ERR("%s: invalid json header in '%s': %s\n", __func__, path.c_str(), e.what());
        return false;
    }
    const size_t base = 8 + n_header;
    if (data_start) {
        *data_start = base;
    }
    try {
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") {
                if (metadata) {
                    *metadata = it.value().dump();
                }
                continue;
            }
            const auto & v = it.value();
            mg_tensor_desc d;
            d.name = it.key();
            d.type = st_dtype_to_ggml(v.at("dtype").get<std::string>());
            if (d.type == GGML_TYPE_COUNT) {
                MG_ERR("%s: unsupported dtype '%s' for tensor '%s'\n", __func__, v.at("dtype").get<std::string>().c_str(), d.name.c_str());
                return false;
            }
            std::vector<int64_t> shape = v.at("shape").get<std::vector<int64_t>>();
            d.ne.assign(shape.rbegin(), shape.rend());
            if (d.ne.empty()) {
                d.ne.push_back(1);
            }
            const auto offs = v.at("data_offsets").get<std::vector<size_t>>();
            if (!mg_desc_finalize(d) || offs.size() != 2 || offs[1] < offs[0] || offs[1] - offs[0] != d.nbytes) {
                MG_ERR("%s: invalid shape or data_offsets for tensor '%s'\n", __func__, d.name.c_str());
                return false;
            }
            d.offset = base + offs[0];
            out.push_back(std::move(d));
        }
    } catch (const std::exception & e) {
        MG_ERR("%s: invalid safetensors header in '%s': %s\n", __func__, path.c_str(), e.what());
        return false;
    }
    return true;
}

//
// gguf
//

namespace {
struct gguf_reader {
    std::ifstream & f;
    bool ok = true;
    explicit gguf_reader(std::ifstream & f) : f(f) {}
    template <typename T> T read() {
        T v{};
        f.read((char *) &v, sizeof(v));
        if (!f) ok = false;
        return v;
    }
    std::string read_str() {
        const uint64_t n = read<uint64_t>();
        if (!ok || n > (1u << 24)) { ok = false; return ""; }
        std::string s(n, '\0');
        f.read(s.data(), n);
        if (!f) ok = false;
        return s;
    }
    // returns the value as a string if it is a string, skips it otherwise
    std::string read_value(uint32_t type, bool * is_str) {
        if (is_str) *is_str = false;
        switch (type) {
            case GGUF_TYPE_UINT8:   read<uint8_t>();  break;
            case GGUF_TYPE_INT8:    read<int8_t>();   break;
            case GGUF_TYPE_UINT16:  read<uint16_t>(); break;
            case GGUF_TYPE_INT16:   read<int16_t>();  break;
            case GGUF_TYPE_UINT32:  read<uint32_t>(); break;
            case GGUF_TYPE_INT32:   read<int32_t>();  break;
            case GGUF_TYPE_FLOAT32: read<float>();    break;
            case GGUF_TYPE_BOOL:    read<uint8_t>();  break;
            case GGUF_TYPE_UINT64:  read<uint64_t>(); break;
            case GGUF_TYPE_INT64:   read<int64_t>();  break;
            case GGUF_TYPE_FLOAT64: read<double>();   break;
            case GGUF_TYPE_STRING:  if (is_str) *is_str = true; return read_str();
            case GGUF_TYPE_ARRAY: {
                const uint32_t et = read<uint32_t>();
                const uint64_t n  = read<uint64_t>();
                for (uint64_t i = 0; i < n && ok; i++) {
                    read_value(et, nullptr);
                }
                break;
            }
            default: ok = false;
        }
        return "";
    }
};
}

bool mg_gguf_read_header(const std::string & path, std::vector<mg_tensor_desc> & out, std::map<std::string, std::string> * kv_str) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        MG_ERR("%s: failed to open '%s'\n", __func__, path.c_str());
        return false;
    }
    gguf_reader r(fin);
    char magic[4];
    fin.read(magic, 4);
    if (!fin || memcmp(magic, "GGUF", 4) != 0) {
        MG_ERR("%s: '%s' is not a gguf file\n", __func__, path.c_str());
        return false;
    }
    const uint32_t version   = r.read<uint32_t>();
    const uint64_t n_tensors = r.read<uint64_t>();
    const uint64_t n_kv      = r.read<uint64_t>();
    if (!r.ok || version < 2 || n_tensors > (1u << 24) || n_kv > (1u << 24)) {
        MG_ERR("%s: invalid gguf header in '%s'\n", __func__, path.c_str());
        return false;
    }
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
    for (uint64_t i = 0; i < n_kv && r.ok; i++) {
        const std::string key = r.read_str();
        const uint32_t type = r.read<uint32_t>();
        if (key == "general.alignment" && type == GGUF_TYPE_UINT32) {
            alignment = r.read<uint32_t>();
            if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
                MG_ERR("%s: invalid general.alignment %u in '%s'\n", __func__, alignment, path.c_str());
                return false;
            }
            continue;
        }
        bool is_str = false;
        std::string val = r.read_value(type, &is_str);
        if (is_str && kv_str) {
            (*kv_str)[key] = std::move(val);
        }
    }
    std::vector<mg_tensor_desc> descs;
    for (uint64_t i = 0; i < n_tensors && r.ok; i++) {
        mg_tensor_desc d;
        d.name = r.read_str();
        const uint32_t n_dims = r.read<uint32_t>();
        if (n_dims > 8) { r.ok = false; break; }
        for (uint32_t k = 0; k < n_dims; k++) {
            d.ne.push_back((int64_t) r.read<uint64_t>());
        }
        d.type   = (ggml_type) r.read<uint32_t>();
        d.offset = r.read<uint64_t>();
        if (!r.ok || !mg_desc_finalize(d)) { r.ok = false; break; }
        descs.push_back(std::move(d));
    }
    if (!r.ok) {
        MG_ERR("%s: failed to parse gguf header of '%s'\n", __func__, path.c_str());
        return false;
    }
    const size_t header_end = (size_t) fin.tellg();
    const size_t data_start = (header_end + alignment - 1) / alignment * alignment;
    for (auto & d : descs) {
        d.offset += data_start;
    }
    out.insert(out.end(), descs.begin(), descs.end());
    return true;
}

//
// weight loading
//

static void convert_row(ggml_type src, ggml_type dst, const uint8_t * in, uint8_t * out, int64_t n) {
    if (src == dst) {
        memcpy(out, in, ggml_row_size(src, n));
        return;
    }
    std::vector<float> tmp(n);
    if (src == GGML_TYPE_BF16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) in, tmp.data(), n);
    } else if (src == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) in, tmp.data(), n);
    } else if (src == GGML_TYPE_F32) {
        memcpy(tmp.data(), in, n * sizeof(float));
    } else {
        GGML_ABORT("unsupported conversion");
    }
    if (dst == GGML_TYPE_F32) {
        memcpy(out, tmp.data(), n * sizeof(float));
    } else if (dst == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(tmp.data(), (ggml_fp16_t *) out, n);
    } else {
        GGML_ABORT("unsupported conversion");
    }
}

bool mg_weights_load(mg_weights & w, const std::string & path, ggml_backend_buffer_type_t buft, const mg_load_opts & opts) {
    std::vector<mg_tensor_desc> descs;
    bool is_gguf = false;
    {
        std::ifstream fin(path, std::ios::binary);
        char magic[4] = {0};
        fin.read(magic, 4);
        is_gguf = memcmp(magic, "GGUF", 4) == 0;
    }
    if (is_gguf) {
        if (!mg_gguf_read_header(path, descs, nullptr)) {
            return false;
        }
    } else {
        if (!mg_safetensors_read_header(path, descs, nullptr, nullptr)) {
            return false;
        }
    }

    // plan: (dst tensor, src desc, tap index or -1)
    struct plan_item {
        ggml_tensor *  dst;
        mg_tensor_desc src;
        int            tap;  // -1 = whole tensor
        int            n_taps;
        std::string    name; // full (renamed) name, ggml truncates t->name
    };
    std::vector<plan_item> plan;

    if (w.buf) {
        MG_ERR("%s: weights already loaded\n", __func__);
        return false;
    }
    {
        size_t n_tensors = 0;
        for (const auto & d : descs) {
            n_tensors += d.ne.size() == 5 ? (size_t) std::max<int64_t>(d.ne[2], 1) : 1;
        }
        ggml_init_params ip = {
            /*.mem_size   =*/ (n_tensors + 16) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        w.ctx.reset(ggml_init(ip));
    }
    ggml_context * ctx = w.ctx.get();

    for (auto & d : descs) {
        if (opts.keep && !opts.keep(d)) {
            continue;
        }
        std::string name = opts.rename ? opts.rename(d.name) : d.name;
        if (name.empty()) {
            continue;
        }
        ggml_type type = d.type;
        if (type == GGML_TYPE_BF16 && opts.bf16_to_f16) {
            type = (d.ne.size() == 1 && opts.f32_from_bf16_small) ? GGML_TYPE_F32 : GGML_TYPE_F16;
        }
        if (d.ne.size() == 5) {
            if (!opts.split_conv3d) {
                MG_ERR("%s: tensor '%s' has 5 dims\n", __func__, d.name.c_str());
                return false;
            }
            // [KW, KH, KT, IC, OC] -> KT x [KW, KH, IC, OC]
            const int n_taps = (int) d.ne[2];
            for (int k = 0; k < n_taps; k++) {
                ggml_tensor * t = ggml_new_tensor_4d(ctx, type, d.ne[0], d.ne[1], d.ne[3], d.ne[4]);
                std::string tname = mg_format("%s.t%d", name.c_str(), k);
                ggml_set_name(t, tname.c_str());
                w.tensors[tname] = t;
                plan.push_back({ t, d, k, n_taps, tname });
            }
            continue;
        }
        if (d.ne.size() > 4) {
            MG_ERR("%s: tensor '%s' has too many dims\n", __func__, d.name.c_str());
            return false;
        }
        ggml_tensor * t = ggml_new_tensor(ctx, type, (int) d.ne.size(), d.ne.data());
        ggml_set_name(t, name.c_str());
        w.tensors[name] = t;
        plan.push_back({ t, d, -1, 1, name });
    }

    // allocate
    {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            MG_ERR("%s: failed to allocate buffer for '%s'\n", __func__, path.c_str());
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        w.buf.reset(buf);
    }

    // read data
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        MG_ERR("%s: failed to open '%s'\n", __func__, path.c_str());
        return false;
    }
    std::vector<uint8_t> read_buf;
    std::vector<uint8_t> tap_buf;
    std::vector<uint8_t> conv_buf;
    for (auto & p : plan) {
        const mg_tensor_desc & d = p.src;
        ggml_tensor * t = p.dst;
        const size_t  nbytes = ggml_nbytes(t);
        const int64_t nelem  = ggml_nelements(t);

        read_buf.resize(d.nbytes);
        fin.seekg(d.offset);
        fin.read((char *) read_buf.data(), d.nbytes);
        if (!fin) {
            MG_ERR("%s: failed to read tensor '%s'\n", __func__, d.name.c_str());
            return false;
        }

        uint8_t * data = read_buf.data();
        if (p.tap >= 0) {
            // strided copy of one temporal tap
            const int64_t KW = d.ne[0], KH = d.ne[1], KT = d.ne[2], IC = d.ne[3], OC = d.ne[4];
            const size_t src_ts = ggml_type_size(d.type);
            const size_t blk = KW * KH;
            tap_buf.resize(blk * IC * OC * src_ts);
            for (int64_t oc = 0; oc < OC; oc++) {
                for (int64_t ic = 0; ic < IC; ic++) {
                    const size_t src_off = (((oc * IC + ic) * KT + p.tap) * blk) * src_ts;
                    const size_t dst_off = ((oc * IC + ic) * blk) * src_ts;
                    memcpy(tap_buf.data() + dst_off, read_buf.data() + src_off, blk * src_ts);
                }
            }
            data = tap_buf.data();
        }
        if (t->type != d.type) {
            conv_buf.resize(nbytes);
            convert_row(d.type, t->type, data, conv_buf.data(), nelem);
            data = conv_buf.data();
        }
        if (opts.transform && t->type == GGML_TYPE_F32) {
            opts.transform(p.name, (float *) data, nelem);
        }
        ggml_backend_tensor_set(t, data, 0, nbytes);
    }
    MG_INF("%s: loaded %zu tensors (%.1f MiB) from %s\n", __func__, plan.size(), w.size() / 1024.0 / 1024.0, path.c_str());
    return true;
}

//
// backend
//

bool mg_backend::init(bool use_gpu, int n_threads_, size_t max_nodes_) {
    n_threads = n_threads_;
    max_nodes = max_nodes_;
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        MG_ERR("%s: failed to initialize CPU backend\n", __func__);
        return false;
    }
    if (use_gpu) {
        gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!gpu) {
            gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
    }
    if (gpu) {
        MG_INF("%s: using %s backend\n", __func__, ggml_backend_name(gpu));
        ptrs.push_back(gpu);
        bufts.push_back(ggml_backend_get_default_buffer_type(gpu));
    } else {
        MG_INF("%s: using CPU backend\n", __func__);
    }
    ptrs.push_back(cpu);
    bufts.push_back(ggml_backend_get_default_buffer_type(cpu));
    {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
        auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads) {
            set_n_threads(cpu, n_threads);
        }
    }
    release_compute();
    return sched != nullptr;
}

void mg_backend::release_compute() {
    sched.reset(ggml_backend_sched_new(ptrs.data(), bufts.data(), (int) ptrs.size(), max_nodes, false, true));
}

void mg_backend::free() {
    sched.reset();
    if (gpu) {
        ggml_backend_free(gpu);
        gpu = nullptr;
    }
    if (cpu) {
        ggml_backend_free(cpu);
        cpu = nullptr;
    }
}

bool mg_backend::supports_op(const ggml_tensor * op) const {
    ggml_backend_t b = gpu ? gpu : cpu;
    return ggml_backend_supports_op(b, op);
}

//
// graph
//

mg_graph::mg_graph(const mg_backend & be) {
    const size_t max_nodes = be.max_nodes;
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx.reset(ggml_init(ip));
    gf = ggml_new_graph_custom(ctx.get(), max_nodes, false);
}

ggml_tensor * mg_graph::new_input(ggml_type type, const std::vector<int64_t> & ne, const void * data, const char * name) {
    ggml_tensor * t = ggml_new_tensor(ctx.get(), type, (int) ne.size(), ne.data());
    ggml_set_name(t, name);
    ggml_set_input(t);
    input in;
    in.t = t;
    in.data.resize(ggml_nbytes(t));
    memcpy(in.data.data(), data, in.data.size());
    inputs.push_back(std::move(in));
    return t;
}

void mg_graph::mark_output(ggml_tensor * t) {
    ggml_set_output(t);
    ggml_build_forward_expand(gf, t);
    outputs.push_back(t);
}

bool mg_graph::compute(mg_backend & be) {
    ggml_backend_sched_reset(be.sched.get());
    if (!ggml_backend_sched_alloc_graph(be.sched.get(), gf)) {
        MG_ERR("%s: failed to allocate graph (%d nodes)\n", __func__, ggml_graph_n_nodes(gf));
        return false;
    }
    for (auto * b : be.ptrs) {
        const size_t sz = ggml_backend_sched_get_buffer_size(be.sched.get(), b);
        if (sz > 0) {
            MG_DBG("%s: %s compute buffer %.1f MiB (%d nodes)\n", __func__, ggml_backend_name(b), sz / 1024.0 / 1024.0, ggml_graph_n_nodes(gf));
        }
    }
    for (auto & in : inputs) {
        ggml_backend_tensor_set(in.t, in.data.data(), 0, in.data.size());
    }
    ggml_status st = ggml_backend_sched_graph_compute(be.sched.get(), gf);
    if (st != GGML_STATUS_SUCCESS) {
        MG_ERR("%s: graph compute failed with status %d\n", __func__, (int) st);
        return false;
    }
    return true;
}

void mg_graph::get_output(ggml_tensor * t, std::vector<float> & out) const {
    GGML_ASSERT(t->type == GGML_TYPE_F32);
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

//
// helpers
//

ggml_tensor * mg_modulate(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * shift) {
    // x * (1 + scale) + shift
    ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, x, scale), x);
    if (shift) {
        y = ggml_add(ctx, y, shift);
    }
    return y;
}

ggml_tensor * mg_linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

void mg_dump(const char * name, const std::vector<float> & data) {
    const char * prefix = getenv("MEDIAGEN_DUMP");
    if (!prefix) {
        return;
    }
    std::string path = mg_format("%s.%s.bin", prefix, name);
    FILE * f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(data.data(), sizeof(float), data.size(), f);
        fclose(f);
        MG_INF("%s: wrote %zu floats to %s\n", __func__, data.size(), path.c_str());
    }
}
