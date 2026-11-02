// loads a GGUF through the regular model path and checks that ggml_backend_tensor_get returns the file bytes
// usage: test-model-tensor-roundtrip <model.gguf> [device-name] [type,type,...]
#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"

#include "../src/llama-model.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static long vm_hwm_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            return std::stol(line.substr(6));
        }
    }
    return -1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [device-name] [type,type,...]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string dev_name = argc > 2 ? argv[2] : "";
    std::set<std::string> want;
    if (argc > 3) {
        std::stringstream ss(argv[3]);
        for (std::string s; std::getline(ss, s, ',');) {
            want.insert(s);
        }
    }

    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gguf = gguf_init_from_file(path.c_str(), gp);
    if (!gguf) {
        fprintf(stderr, "failed to read %s\n", path.c_str());
        return 2;
    }
    const size_t data_off = gguf_get_data_offset(gguf);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    std::vector<ggml_backend_dev_t> devs;
    ggml_backend_dev_t dev = nullptr;
    if (!dev_name.empty()) {
        ggml_backend_dev_t d = ggml_backend_dev_by_name(dev_name.c_str());
        if (!d) {
            fprintf(stderr, "device %s not found\n", dev_name.c_str());
            return 2;
        }
        dev  = d;
        devs = { d, nullptr };
        mp.devices = devs.data();
    }

    const long hwm_before = vm_hwm_kb();
    llama_model * model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 2;
    }
    const long hwm_after = vm_hwm_kb();

    std::ifstream file(path, std::ios::binary);
    std::map<std::string, std::vector<int>> stat;  // "type buffer" -> {checked, bad}
    int n_bad = 0;

    for (const auto & [name, t] : model->tensors_by_name) {
        const char * tname = ggml_type_name(t->type);
        if (!ggml_is_quantized(t->type) || (!want.empty() && !want.count(tname))) {
            continue;
        }
        const int64_t id = gguf_find_tensor(gguf, name.c_str());
        if (id < 0) {
            continue;
        }
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> ref(nbytes), got(nbytes);
        file.seekg(data_off + gguf_get_tensor_offset(gguf, id));
        file.read((char *) ref.data(), nbytes);
        std::string where = t->buffer ? ggml_backend_buffer_name(t->buffer) : "none";

        if (dev && t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
            // weight stayed on the host (op not supported yet): push the file bytes through the device buffer instead
            ggml_init_params ip = { ggml_tensor_overhead()*2, nullptr, true };
            ggml_context * c = ggml_init(ip);
            ggml_tensor * d = ggml_new_tensor(c, t->type, GGML_MAX_DIMS, t->ne);
            ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), ggml_backend_buft_get_alloc_size(ggml_backend_dev_buffer_type(dev), d));
            if (!b || ggml_backend_tensor_alloc(b, d, ggml_backend_buffer_get_base(b)) != GGML_STATUS_SUCCESS) {
                printf("ALLOC FAILED %s\n", name.c_str());
                n_bad++;
                continue;
            }
            ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_tensor_set(d, ref.data(), 0, nbytes);
            ggml_backend_tensor_get(d, got.data(), 0, nbytes);
            where = std::string("pushed to ") + ggml_backend_buffer_name(b);
            ggml_backend_buffer_free(b);
            ggml_free(c);
        } else {
            ggml_backend_tensor_get(t, got.data(), 0, nbytes);
        }

        const bool ok = memcmp(ref.data(), got.data(), nbytes) == 0;
        auto & s = stat[std::string(tname) + " @ " + where];
        s.resize(2);
        s[0]++;
        if (!ok) {
            s[1]++;
            n_bad++;
            printf("MISMATCH %s (%s %ldx%ldx%ld)\n", name.c_str(), tname, (long) t->ne[0], (long) t->ne[1], (long) t->ne[2]);
        }
    }

    for (const auto & [k, v] : stat) {
        printf("%-32s checked=%d mismatched=%d\n", k.c_str(), v[0], v[1]);
    }
    printf("VmHWM before load: %ld kB, after load: %ld kB, final: %ld kB\n", hwm_before, hwm_after, vm_hwm_kb());
    printf("%s\n", stat.empty() ? "NOTHING CHECKED" : (n_bad ? "FAILED" : "PASSED"));

    llama_model_free(model);
    gguf_free(gguf);
    return n_bad || stat.empty() ? 1 : 0;
}
