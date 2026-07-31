#include "tessera-coreml.h"
#include "tessera-coreml-builder.h"
#include "tessera-coreml-metadata.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

static bool file_exists(const char * path) {
    std::ifstream f(path);
    return f.good();
}

int main() {
    int pass = 0, fail = 0;
    auto check = [&](bool ok, const char * name) {
        if (ok) { printf("  [coreml] %s: PASS\n", name); pass++; }
        else    { printf("  [coreml] %s: FAIL\n", name); fail++; }
    };

    // --- stock ops validation ---
    {
        ts_coreml_tensor_desc t[2];
        t[0] = {"attn_q", 4096, 4096, 43, false};  // TESSERA_T640
        t[1] = {"ffn_up",  14336, 4096, 43, true};
        std::string err;
        check(ts_coreml_validate_stock_ops(t, 2, &err) == 0, "validate_stock_ops_2d");

        t[1].ggml_type = 44; // TESSERA_T640_3D
        check(ts_coreml_validate_stock_ops(t, 2, &err) == -1, "validate_rejects_3d");
    }

    // --- spec generation ---
    {
        ts_coreml_tensor_desc t[1];
        t[0] = {"test_layer", 64, 32, 43, false};
        ts_coreml_params p = ts_coreml_default_params();
        p.output_dir = "/tmp/tessera_coreml_test";
        std::string err;
        int rc = ts_coreml_generate_spec(t, 1, &p, "/tmp/tessera_coreml_spec.json", &err);
        check(rc == 0, "generate_spec");
        check(file_exists("/tmp/tessera_coreml_spec.json"), "spec_file_exists");
    }

    // --- package builder ---
    {
        const int64_t O = 8, I = 16;
        std::vector<uint16_t> w(O * I, 0x3c00); // fp16(1.0)
        std::vector<uint16_t> as(I, 0x3c00);

        ts_coreml_builder_tensor t;
        t.name = "test_weight";
        t.out_dim = O;
        t.in_dim = I;
        t.weights_f16 = w.data();
        t.has_act_scale = true;
        t.act_scale_f16 = as.data();

        ts_coreml_builder_params p;
        p.output_path = "/tmp/tessera_coreml_test.mlpackage";
        p.model_name = "test_model";
        p.compute_units = 0;

        std::string err;
        int rc = ts_coreml_build_package(&t, 1, &p, &err);
        check(rc == 0, "build_package");
        check(file_exists("/tmp/tessera_coreml_test.mlpackage/Manifest.json"), "manifest_exists");
        check(file_exists("/tmp/tessera_coreml_test.mlpackage/Data/model.mlmodel"), "model_exists");
        check(file_exists("/tmp/tessera_coreml_test.mlpackage/Data/weights_0.bin"), "weights_exist");
        check(file_exists("/tmp/tessera_coreml_test.mlpackage/Data/act_scale_0.bin"), "act_scale_exists");
        check(file_exists("/tmp/tessera_coreml_test.mlpackage/Metadata/model.json"), "metadata_exists");

        check(ts_coreml_total_weight_bytes(&t, 1) == O * I * 2 + I * 2, "total_weight_bytes");
    }

    // --- metadata: sidecar round-trip ---
    {
        // Write a test sidecar
        {
            std::ofstream f("/tmp/tessera_coreml_sidecar.json");
            f << "{\"n_layers\": 32, \"hidden_dim\": 4096, \"n_heads\": 32, "
              << "\"model_name\": \"test-model\", \"calibration_mse\": 0.001, "
              << "\"has_act_scale\": true, \"quant_type\": 43}\n";
        }
        ts_coreml_config cfg;
        std::string err;
        int rc = ts_coreml_config_from_sidecar("/tmp/tessera_coreml_sidecar.json", &cfg, &err);
        check(rc == 0, "sidecar_read");
        check(cfg.n_layers == 32, "sidecar_n_layers");
        check(cfg.hidden_dim == 4096, "sidecar_hidden_dim");
        check(cfg.model_name == "test-model", "sidecar_model_name");
        check(cfg.has_act_scale == true, "sidecar_has_act_scale");

        // Merge with empty GGUF config
        ts_coreml_config gguf_cfg = {};
        ts_coreml_config merged;
        std::vector<std::string> warnings;
        rc = ts_coreml_config_merge(&gguf_cfg, &cfg, &merged, &warnings);
        check(rc == 0, "merge_ok");
        check(merged.n_layers == 32, "merge_n_layers");
        check(warnings.empty(), "merge_no_warnings");
    }

    // --- metadata: sidecar not found ---
    {
        ts_coreml_config cfg;
        std::string err;
        check(ts_coreml_config_from_sidecar("/tmp/nonexistent_sidecar.json", &cfg, &err) == 1,
              "sidecar_not_found");
    }

    printf("\ncoreml_bridge: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
