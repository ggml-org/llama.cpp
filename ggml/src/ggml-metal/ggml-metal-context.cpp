#include "ggml-metal-context.h"

#include <string>
#include <unordered_map>

struct ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> data;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(void) {
    ggml_metal_pipelines_t res = new ggml_metal_pipelines();

    return res;
}

void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    for (auto it = ppls->data.begin(); it != ppls->data.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

void ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline) {
    ppls->data[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name) {
    if  (ppls->data.find(name) == ppls->data.end()) {
        return nullptr;
    }

    return ppls->data[name];
}
