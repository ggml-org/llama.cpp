#include "tessera-coreml.h"

#include "ggml.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static ts_coreml_tensor_desc make_desc(const char * name, int32_t ggml_type) {
    ts_coreml_tensor_desc t;
    t.name          = name;
    t.out_dim       = 4096;
    t.in_dim        = 4096;
    t.ggml_type     = ggml_type;
    t.has_act_scale = true;
    return t;
}

int main() {
    // 1. CoreML is available on macOS
    assert(ts_coreml_available() == true);

    // 2. all-T640 validates clean (stock ops v1)
    ts_coreml_tensor_desc stock[3] = {
        make_desc("blk.0.attn_q.weight", GGML_TYPE_TESSERA_T640),
        make_desc("blk.0.ffn_gate.weight", GGML_TYPE_TESSERA_T640),
        make_desc("blk.0.ffn_down.weight", GGML_TYPE_TESSERA_T640),
    };
    std::string err;
    assert(ts_coreml_validate_stock_ops(stock, 3, &err) == 0);
    assert(err.empty());

    // 3. a T640_3D tensor needs custom ops (v2) -> rejected
    ts_coreml_tensor_desc expert = make_desc("blk.0.expert_bank", GGML_TYPE_TESSERA_T640_3D);
    err.clear();
    assert(ts_coreml_validate_stock_ops(&expert, 1, &err) == -1);
    assert(err.find("TESSERA_T640_3D") != std::string::npos);

    // 4. generate spec, verify the file exists and is non-empty
    ts_coreml_params params = ts_coreml_default_params();
    params.output_dir = "/tmp/test_coreml_out";

    const char * spec_path = "/tmp/test_coreml_spec.json";
    err.clear();
    assert(ts_coreml_generate_spec(stock, 3, &params, spec_path, &err) == 0);
    assert(err.empty());

    std::ifstream f(spec_path, std::ios::binary);
    assert(f.good());
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string spec = ss.str();
    assert(!spec.empty());
    assert(spec.find("\"schema\": \"tessera.coreml.spec.v1\"") != std::string::npos);
    assert(spec.find("TESSERA_T640") != std::string::npos);
    assert(spec.find("\"n_tensors\": 3") != std::string::npos);

    // 5. spec generation refuses custom-op tensors
    err.clear();
    assert(ts_coreml_generate_spec(&expert, 1, &params, spec_path, &err) == -1);
    assert(!err.empty());

    printf("PASS\n");
    return 0;
}
