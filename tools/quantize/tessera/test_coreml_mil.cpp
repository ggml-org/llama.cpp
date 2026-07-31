#include "tessera-coreml.h"
#include "tessera-coreml-builder.h"
#include "tessera-coreml-mil.h"
#include "tessera-coreml-telemetry.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool file_exists(const char * path) {
    std::ifstream f(path);
    return f.good();
}

static long file_size(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        return -1;
    }
    return (long) f.tellg();
}

static std::string read_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    int pass = 0, fail = 0;
    auto check = [&](bool ok, const char * name) {
        if (ok) { printf("  [coreml-mil] %s: PASS\n", name); pass++; }
        else    { printf("  [coreml-mil] %s: FAIL\n", name); fail++; }
    };

    // --- MIL builder: const -> matmul graph, JSON structure ---
    {
        ts_mil_builder b;
        ts_mil_builder_init(&b, "main");

        const int64_t in_shape[2]  = {1, 4};
        const int64_t w_shape[2]   = {8, 4};
        ts_mil_add_input(&b, "x", TS_MIL_FP16, in_shape, 2);
        std::string w = ts_mil_const(&b, "w", TS_MIL_FP16, w_shape, 2, "w.fp16.bin");
        std::string y = ts_mil_matmul(&b, "x", w.c_str(), true);
        ts_mil_add_output(&b, y.c_str());

        std::string err;
        check(ts_mil_build(&b, &err) == 0, "mil_build_ok");

        check(ts_mil_emit_json(&b, "/tmp/tessera_mil_graph.json", &err) == 0, "mil_emit_json");
        const std::string json = read_file("/tmp/tessera_mil_graph.json");
        check(json.find("\"blockSpecializations\"") != std::string::npos, "mil_json_has_blocks");
        check(json.find("\"CoreML9\"") != std::string::npos, "mil_json_opset");
        check(json.find("\"type\": \"matmul\"") != std::string::npos, "mil_json_matmul");
        check(json.find("\"type\": \"const\"") != std::string::npos, "mil_json_const");
        check(json.find("@w.fp16.bin") != std::string::npos, "mil_json_weight_ref");
        check(json.find("\"transpose_y\": \"true\"") != std::string::npos, "mil_json_transpose_y");
        // matmul shape inference: [1,4] @ [8,4]^T -> [1,8]
        check(json.find("[1, 8]") != std::string::npos, "mil_matmul_shape_inference");
    }

    // --- MIL builder: v2 custom op (tessera_t640_dequant) ---
    {
        ts_mil_builder b;
        ts_mil_builder_init(&b, "main");
        const int64_t packed_shape[2] = {8, 32};
        const int64_t scale_shape[2]  = {8, 1};
        const int64_t off_shape[1]    = {9};
        const int64_t out_shape[2]    = {8, 640};
        std::string p  = ts_mil_const(&b, "packed",  TS_MIL_UINT8, packed_shape, 2, "p.bin");
        std::string ps = ts_mil_const(&b, "pages",   TS_MIL_FP16,  scale_shape,  2, "ps.bin");
        std::string ls = ts_mil_const(&b, "lanes",   TS_MIL_INT8,  scale_shape,  2, "ls.bin");
        std::string oo = ts_mil_const(&b, "offs",    TS_MIL_INT32, off_shape,    1, "oo.bin");
        std::string oc = ts_mil_const(&b, "ocols",   TS_MIL_INT32, off_shape,    1, "oc.bin");
        std::string ov = ts_mil_const(&b, "ovals",   TS_MIL_FP16,  off_shape,    1, "ov.bin");
        std::string as = ts_mil_const(&b, "act",     TS_MIL_FP16,  off_shape,    1, "as.bin");
        std::string y  = ts_mil_tessera_dequant(&b, p.c_str(), ps.c_str(), ls.c_str(),
                                                oo.c_str(), oc.c_str(), ov.c_str(),
                                                as.c_str(), out_shape, 2);
        ts_mil_add_output(&b, y.c_str());

        std::string err;
        check(ts_mil_build(&b, &err) == 0, "mil_custom_op_build_ok");
        const std::string json = ts_mil_to_json(&b);
        check(json.find("tessera_t640_dequant") != std::string::npos, "mil_custom_op_type");
        check(json.find("\"page_size\": \"640\"") != std::string::npos, "mil_custom_op_page_size");
        check(json.find("\"lanes_per_page\": \"32\"") != std::string::npos, "mil_custom_op_lanes");
    }

    // --- MIL builder: SSA validation catches undefined reference ---
    {
        ts_mil_builder b;
        ts_mil_builder_init(&b, "main");
        const int64_t shape[2] = {1, 4};
        ts_mil_add_input(&b, "x", TS_MIL_FP16, shape, 2);
        ts_mil_matmul(&b, "x", "does_not_exist", false);
        std::string y = "x";
        ts_mil_add_output(&b, y.c_str());
        std::string err;
        check(ts_mil_build(&b, &err) == -1, "mil_ssa_undefined_ref");
        check(err.find("does_not_exist") != std::string::npos, "mil_ssa_err_names_value");
    }

    // --- MIL builder: missing block output rejected ---
    {
        ts_mil_builder b;
        ts_mil_builder_init(&b, "main");
        const int64_t shape[2] = {1, 4};
        ts_mil_add_input(&b, "x", TS_MIL_FP16, shape, 2);
        std::string err;
        check(ts_mil_build(&b, &err) == -1, "mil_no_outputs_rejected");
    }

    // --- Weight dequant: hand-encoded Tile640 cluster matches ggml algorithm ---
    {
        // one row, in_dim = 20 (exactly one lane), pages = 1
        const int64_t out_dim = 1, in_dim = 20;
        std::vector<uint32_t> packed(32, 0);
        // lane 0, group 0 trits [1,2,0,1,2] -> group_val = 1 + 6 + 0 + 27 + 162 = 196
        packed[0] = 196;
        std::vector<uint16_t> page_scales(1, 0x3c00);   // fp16(1.0)
        std::vector<int8_t>   lane_scales(32, 0);
        lane_scales[0] = 127;                            // scale = 1.0 * 127/127 = 1.0

        ts_coreml_weight_src src = {};
        src.name        = "test.weight";
        src.out_dim     = out_dim;
        src.in_dim      = in_dim;
        src.packed      = (const uint8_t *) packed.data();
        src.page_scales = (const uint8_t *) page_scales.data();
        src.lane_scales = (const uint8_t *) lane_scales.data();

        std::vector<float> y(out_dim * in_dim, 999.0f);
        ts_coreml_dequant_t640(&src, y.data());

        const float expect[20] = {1, -1, 0, 1, -1, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        bool ok = true;
        for (int i = 0; i < 20; i++) {
            if (y[i] != expect[i]) {
                ok = false;
            }
        }
        check(ok, "dequant_trit_pattern");
    }

    // --- Weight serialization: fp16 blob size + values ---
    {
        const int64_t out_dim = 1, in_dim = 20;
        std::vector<uint32_t> packed(32, 0);
        packed[0] = 196;
        std::vector<uint16_t> page_scales(1, 0x3c00);
        std::vector<int8_t>   lane_scales(32, 0);
        lane_scales[0] = 127;

        ts_coreml_weight_src src = {};
        src.name        = "blk.0.attn_q.weight";   // exercises name sanitization
        src.out_dim     = out_dim;
        src.in_dim      = in_dim;
        src.packed      = (const uint8_t *) packed.data();
        src.page_scales = (const uint8_t *) page_scales.data();
        src.lane_scales = (const uint8_t *) lane_scales.data();

        std::string err;
        ts_coreml_weight_out out;
        int rc = ts_coreml_serialize_weights(&src, "/tmp", false, &out, &err);
        check(rc == 0, "serialize_weights_ok");
        check(out.custom_op == false, "serialize_weights_v1");
        check(out.n_bytes == out_dim * in_dim * 2, "serialize_weights_size");

        const std::string path = "/tmp/" + out.blob_name;
        check(file_exists(path.c_str()), "serialize_weights_blob_exists");
        check(file_size(path.c_str()) == out_dim * in_dim * 2, "serialize_weights_file_size");
        // blob name is sanitized (dots -> underscores) and carries the fp16 suffix
        check(out.blob_name == "blk_0_attn_q_weight.fp16.bin", "serialize_weights_blob_name");

        // read back and verify the first three fp16 values: +1, -1, 0
        std::ifstream f(path, std::ios::binary);
        std::vector<uint16_t> blob(out_dim * in_dim);
        f.read((char *) blob.data(), blob.size() * 2);
        check(blob[0] == 0x3c00, "serialize_weights_val_pos");  // fp16(+1)
        check(blob[1] == 0xbc00, "serialize_weights_val_neg");  // fp16(-1)
        check(blob[2] == 0x0000, "serialize_weights_val_zero");
    }

    // --- Weight serialization: v2 custom-op raw blobs ---
    {
        const int64_t out_dim = 2, in_dim = 640;   // one full page
        const int pages = 1;
        std::vector<uint8_t> packed(out_dim * pages * 32 * 4, 0);
        std::vector<uint8_t> page_scales(out_dim * pages * 2, 0);
        std::vector<uint8_t> lane_scales(out_dim * pages * 32, 0);

        ts_coreml_weight_src src = {};
        src.name        = "raw.weight";
        src.out_dim     = out_dim;
        src.in_dim      = in_dim;
        src.packed      = packed.data();
        src.page_scales = page_scales.data();
        src.lane_scales = lane_scales.data();

        std::string err;
        ts_coreml_weight_out out;
        int rc = ts_coreml_serialize_weights(&src, "/tmp", true, &out, &err);
        check(rc == 0, "serialize_weights_v2_ok");
        check(out.custom_op == true, "serialize_weights_v2_flag");
        check(out.blob_name == "raw_weight.packed.bin", "serialize_weights_v2_name");
        check(file_exists(("/tmp/" + out.blob_name).c_str()), "serialize_weights_v2_blob");
    }

    // --- Telemetry: mock samples are plausible, energy accumulates ---
    {
        ts_coreml_telemetry tel;
        ts_coreml_telemetry_config cfg = ts_coreml_telemetry_default_config();
        cfg.enable = true;
        std::string err;
        check(ts_coreml_telemetry_start(&tel, &cfg, &err) == 0, "telemetry_start");

        bool plausible = true;
        const int N = 32;
        for (int i = 0; i < N; i++) {
            ts_coreml_telemetry_sample_t s;
            if (ts_coreml_telemetry_sample(&tel, &s) != 0) {
                plausible = false;
                break;
            }
            if (s.ane_power_mw < 800.0 || s.ane_power_mw > 1600.0) plausible = false;
            if (s.gpu_power_mw < 200.0 || s.gpu_power_mw > 500.0)  plausible = false;
            if (s.cpu_power_mw < 600.0 || s.cpu_power_mw > 1200.0) plausible = false;
            if (s.dram_power_mw < 300.0 || s.dram_power_mw > 500.0) plausible = false;
            if (s.thermal_state < 0 || s.thermal_state > 3)         plausible = false;
            if (s.battery_current_ma >= 0)                          plausible = false;
        }
        check(plausible, "telemetry_samples_plausible");

        ts_coreml_telemetry_session_energy e;
        ts_coreml_telemetry_session_energy_get(&tel, &e);
        check(e.n_samples == N, "telemetry_sample_count");
        check(e.total_mj > 0.0, "telemetry_energy_accumulates");
        check(e.ane_mj > 0.0 && e.gpu_mj > 0.0 && e.cpu_mj > 0.0, "telemetry_per_rail_energy");

        check(ts_coreml_telemetry_write_summary(&tel, "/tmp/tessera_telemetry.json", &err) == 0,
              "telemetry_write_summary");
        const std::string js = read_file("/tmp/tessera_telemetry.json");
        check(js.find("tessera.coreml.telemetry.v1") != std::string::npos, "telemetry_summary_schema");
        check(js.find("\"source\": \"mock\"") != std::string::npos, "telemetry_summary_mock");
        ts_coreml_telemetry_stop(&tel);

        // sampling after stop fails cleanly
        ts_coreml_telemetry_sample_t s2;
        check(ts_coreml_telemetry_sample(&tel, &s2) == -1, "telemetry_sample_after_stop");
    }

    // --- Compile: error handling (full compile needs the real protobuf build) ---
    {
        std::string err;
        if (ts_coreml_xcrun_available()) {
            // xcrun present: a bogus package path must fail with a clear error
            int rc = ts_coreml_compile("/tmp/does_not_exist.mlpackage", "/tmp", &err);
            check(rc == -1, "compile_bogus_package_fails");
            check(!err.empty(), "compile_bogus_package_err");
        } else {
            // no tools: compile refuses and reports the missing toolchain
            int rc = ts_coreml_compile("/tmp/x.mlpackage", "/tmp", &err);
            check(rc == -1, "compile_no_xcrun_fails");
            check(err.find("xcrun") != std::string::npos, "compile_no_xcrun_err");
        }
    }

    // --- Pipeline: build MIL -> serialize -> write package -> telemetry ---
    {
        const int64_t O = 8, I = 16;
        std::vector<uint16_t> w0(O * I, 0x3c00);
        std::vector<uint16_t> w1(O * I, 0x4000);

        ts_coreml_builder_tensor t[2];
        t[0].name = "blk.0.attn_q.weight";
        t[0].out_dim = O; t[0].in_dim = I;
        t[0].weights_f16 = w0.data();
        t[0].has_act_scale = false; t[0].act_scale_f16 = nullptr;
        t[1].name = "blk.0.ffn_up.weight";
        t[1].out_dim = O; t[1].in_dim = I;
        t[1].weights_f16 = w1.data();
        t[1].has_act_scale = false; t[1].act_scale_f16 = nullptr;

        ts_coreml_convert_params p = ts_coreml_convert_default_params();
        p.output_path = "/tmp/tessera_convert.mlpackage";
        p.model_name  = "convert-test";
        p.compile     = false;   // scaffold model.mlmodel is JSON, not protobuf yet
        p.telemetry   = true;

        ts_coreml_convert_result r;
        std::string err;
        int rc = ts_coreml_convert(t, 2, &p, &r, &err);
        check(rc == 0, "pipeline_convert_ok");
        check(file_exists("/tmp/tessera_convert.mlpackage/Manifest.json"), "pipeline_manifest");
        check(file_exists("/tmp/tessera_convert.mlpackage/Metadata/model.json"), "pipeline_metadata");
        check(file_exists(r.mil_json_path.c_str()), "pipeline_mil_json");

        const std::string mil = read_file(r.mil_json_path.c_str());
        check(mil.find("\"type\": \"matmul\"") != std::string::npos, "pipeline_mil_has_matmul");
        check(mil.find("blk_0_attn_q_weight.fp16.bin") != std::string::npos, "pipeline_mil_weight_ref");

        check(file_exists("/tmp/tessera_convert.mlpackage/Data/blk_0_attn_q_weight.fp16.bin"),
              "pipeline_weight_blob");
        check(file_size("/tmp/tessera_convert.mlpackage/Data/blk_0_ffn_up_weight.fp16.bin") == O * I * 2,
              "pipeline_weight_blob_size");
        check(r.weight_bytes == 2 * O * I * 2, "pipeline_weight_bytes");
        check(r.n_tensors == 2, "pipeline_n_tensors");

        check(!r.telemetry_path.empty() && file_exists(r.telemetry_path.c_str()),
              "pipeline_telemetry_summary");
    }

    printf("\ncoreml_mil: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
