// Unit tests for the --tessera-config INI parser (common/tessera-config.*).
//
// Coverage:
//   1. happy path: parse a small in-memory INI, verify values land in
//      common_tessera_params with the right types.
//   2. parser error paths: malformed line, unclosed section, duplicate
//      key, unknown section, unknown key in a known section, bad enum,
//      bad numeric range.
//   3. end-to-end CLI precedence: write a config file, then call
//      common_params_parse with an explicit CLI flag and verify the CLI
//      value wins over the config value.

#include "arg.h"
#include "common.h"
#include "tessera-args.h"
#include "tessera-config.h"
#include "tessera-debug/tessera-debug.h"
#include "tessera-debug/tessera-matmul-output.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

#undef NDEBUG
#include <cassert>

// Convenience: build a char** argv from a vector of strings for
// common_params_parse. IMPORTANT: this keeps a copy of the strings
// so the c_str() pointers stay valid for the lifetime of the returned
// object. (A naive `for s in v: push_back(s.c_str())` would invalidate
// the pointers when the output vector reallocates.)
struct argv_holder {
    std::vector<std::string> storage;
    std::vector<char *>       ptrs;
    argv_holder(std::vector<std::string> v) : storage(std::move(v)) {
        ptrs.reserve(storage.size());
        for (const auto & s : storage) {
            ptrs.push_back(const_cast<char *>(s.c_str()));
        }
    }
};

// Write `text` to a unique temp file and return the path. Caller deletes.
static std::string write_tmp(const std::string & text) {
    char tmpl[] = "/tmp/tessera-config-XXXXXX";
    int fd = mkstemp(tmpl);
    assert(fd >= 0);
    ssize_t n = write(fd, text.data(), text.size());
    assert(n == (ssize_t) text.size());
    close(fd);
    return std::string(tmpl);
}

static void test_parse_happy(void) {
    printf("test-tessera-config: happy path\n");

    const std::string ini =
        "# simple comment\n"
        "; another comment\n"
        "\n"
        "[evolve]\n"
        "evolve-iters      = 12\n"
        "evolve-islands    = 3\n"
        "evolve-population = 24\n"
        "evolve-seed       = 42\n"
        "evolve-only       = true\n"
        "\n"
        "[awq]\n"
        "awq-alpha = auto\n"
        "awq-clip  = 1.5\n"
        "\n"
        "[policy]\n"
        "outlier-frac      = 0.01\n"
        "range-selection   = septq\n"
        "ternary-threshold = 0.7\n"
        "\n"
        "[w4a4]\n"
        "w4a4                = true\n"
        "w4a4-outlier-thresh = 5.5\n"
        "\n"
        "[l15]\n"
        "l15-dtype = f16\n"
        "\n"
        "[general]\n"
        "mode       = evolve-only\n"
        "nthreads   = 8\n"
        "champq     = on\n"
        "imatrix    = /tmp/test.imatrix\n";

    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    if (!ok) {
        fprintf(stderr, "test-tessera-config: unexpected parse error: %s", err.c_str());
        assert(false);
    }
    assert(cfg.sections.count("evolve") == 1);
    assert(cfg.sections.count("awq") == 1);
    assert(cfg.sections.count("policy") == 1);
    assert(cfg.sections.count("w4a4") == 1);
    assert(cfg.sections.count("l15") == 1);
    assert(cfg.sections.count("general") == 1);

    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    if (!ok) {
        fprintf(stderr, "test-tessera-config: unexpected apply error: %s", err.c_str());
        assert(false);
    }
    assert(p.evolve_iters == 12);
    assert(p.evolve_islands == 3);
    assert(p.evolve_population == 24);
    assert(p.evolve_seed == 42);
    assert(p.evolve_only == true);
    assert(p.awq_alpha == "auto");
    assert(p.awq_clip == 1.5f);
    assert(p.outlier_frac == 0.01f);
    assert(p.range_selection == "septq");
    assert(p.ternary_threshold == "0.7");
    assert(p.w4a4 == true);
    assert(p.w4a4_outlier_thresh == 5.5f);
    // l15-dtype writes to tessera_debug's module global, not to
    // common_tessera_params. Verify via the module getter instead.
    assert(tessera_debug::l15_dtype() == "f16");
    // restore so other tests aren't affected
    tessera_debug::set_l15_dtype("f16");
    assert(p.mode == "evolve-only");
    assert(p.nthreads == 8);
    assert(p.champq == true);
    assert(p.imatrix == "/tmp/test.imatrix");
}

static void test_parse_quoted(void) {
    printf("test-tessera-config: quoted values with embedded escapes\n");

    const std::string ini =
        "[policy]\n"
        "policy = \"/tmp/policy with space.json\"\n"
        "ternary-threshold = '0.5'\n"
        "\n"
        "[anonymize]\n"
        "anonymize-level = light\n"
        "anonymize-map   = \"/tmp/anon map.json\"\n";

    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    if (!ok) {
        fprintf(stderr, "test-tessera-config: %s", err.c_str());
        assert(false);
    }
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    if (!ok) {
        fprintf(stderr, "test-tessera-config: %s", err.c_str());
        assert(false);
    }
    assert(p.policy == "/tmp/policy with space.json");
    assert(p.ternary_threshold == "0.5");
    assert(p.anonymize_level == "light");
    assert(p.anonymize_map == "/tmp/anon map.json");
}

static void test_global_and_general(void) {
    printf("test-tessera-config: global bucket + [general] merge\n");

    // Keys before any [section] header go into the global bucket. The
    // [general] section reuses the same dispatch table, so the two
    // forms are interchangeable.
    const std::string ini =
        "mode = calibrate-only\n"
        "[general]\n"
        "nthreads = 4\n";

    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    if (!ok) { fprintf(stderr, "%s", err.c_str()); assert(false); }

    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    if (!ok) { fprintf(stderr, "%s", err.c_str()); assert(false); }
    assert(p.mode == "calibrate-only");
    assert(p.nthreads == 4);
}

static void test_duplicate_key(void) {
    printf("test-tessera-config: duplicate key in same section is rejected\n");

    const std::string ini =
        "[evolve]\n"
        "evolve-iters = 4\n"
        "evolve-iters = 8\n";

    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == false);
    assert(err.find("duplicate key") != std::string::npos);
    assert(err.find("evolve-iters") != std::string::npos);
    assert(err.find("[evolve]") != std::string::npos);
}

static void test_unclosed_section(void) {
    printf("test-tessera-config: unclosed section header is rejected\n");

    const std::string ini =
        "[evolve\n"
        "evolve-iters = 4\n";

    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == false);
    assert(err.find("unclosed section header") != std::string::npos);
}

static void test_malformed_line(void) {
    printf("test-tessera-config: malformed line is rejected\n");

    // no '=' or ':' separator
    const std::string ini =
        "[evolve]\n"
        "evolve-iters 4\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == false);
    assert(err.find("malformed line") != std::string::npos);
}

static void test_unknown_section(void) {
    printf("test-tessera-config: unknown section is rejected\n");

    const std::string ini =
        "[not-a-section]\n"
        "foo = bar\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == true);  // parser doesn't know the section list
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    assert(ok == false);
    assert(err.find("unknown section") != std::string::npos);
    assert(err.find("[not-a-section]") != std::string::npos);
}

static void test_unknown_key(void) {
    printf("test-tessera-config: unknown key in known section is rejected\n");

    const std::string ini =
        "[evolve]\n"
        "evolve-iters = 4\n"
        "evolve-bogus = 1\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == true);
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    assert(ok == false);
    assert(err.find("unknown key") != std::string::npos);
    assert(err.find("evolve-bogus") != std::string::npos);
}

static void test_bad_enum(void) {
    printf("test-tessera-config: bad enum value is rejected\n");

    const std::string ini =
        "[policy]\n"
        "range-selection = not-a-real-mode\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == true);
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    assert(ok == false);
    assert(err.find("range-selection") != std::string::npos);
}

static void test_out_of_range(void) {
    printf("test-tessera-config: out-of-range numeric value is rejected\n");

    const std::string ini =
        "[policy]\n"
        "outlier-frac = 1.5\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == true);
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    assert(ok == false);
    // Error message names the C++ field; the user-facing key
    // 'outlier-frac' differs from the field 'outlier_frac' only by
    // a hyphen. The error must name one of them.
    assert(err.find("outlier_frac") != std::string::npos ||
           err.find("outlier-frac") != std::string::npos);
}

static void test_bad_int(void) {
    printf("test-tessera-config: non-integer value for int field is rejected\n");

    const std::string ini =
        "[evolve]\n"
        "evolve-iters = not-a-number\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    assert(ok == true);
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    assert(ok == false);
    // Error names the C++ field; user-facing key is the same letters
    // with hyphens instead of underscores.
    assert(err.find("evolve_iters") != std::string::npos ||
           err.find("evolve-iters") != std::string::npos);
}

static void test_load_file_missing(void) {
    printf("test-tessera-config: load of missing file returns false\n");

    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_load("/tmp/this-file-does-not-exist-12345.ini", cfg, err);
    assert(ok == false);
    assert(err.find("failed to open") != std::string::npos);
}

static void test_cli_overrides_config(void) {
    printf("test-tessera-config: CLI flag overrides config-file value\n");

    // Write a config file that sets a few values.
    const std::string ini =
        "[evolve]\n"
        "evolve-iters = 12\n"
        "[general]\n"
        "imatrix = /tmp/from-config.npz\n"
        "nthreads = 8\n";
    const std::string path = write_tmp(ini);

    // argv: --tessera-config <path> --tessera-imatrix /tmp/from-cli.npz
    // (no evolve-iters flag, so the config's 12 should win;
    //  --tessera-imatrix /tmp/from-cli.npz should override the config's
    //  imatrix value). --tessera-mode was removed in Tier 2's hard break,
    //  so we use --tessera-imatrix (top-level TESSERA scope) instead.
    auto argv = argv_holder({
        "binary",
        "--tessera-config", path,
        "--tessera-imatrix", "/tmp/from-cli.npz",
        "-m", "/tmp/dummy.gguf",
    });
    common_params params;
    // --tessera-mode was migrated to LLAMA_EXAMPLE_TESSERA scope by the
    // Tier 1 collapse-parse-paths refactor, so it is no longer visible
    // when parsing with LLAMA_EXAMPLE_COMMON. Use the TESSERA scope here;
    // --tessera-config is in COMMON scope and is still visible because
    // TESSERA inherits COMMON (see common/arg.cpp:1473).
    bool ok = common_params_parse((int) argv.ptrs.size(), argv.ptrs.data(), params, LLAMA_EXAMPLE_TESSERA);
    if (!ok) {
        fprintf(stderr, "test-tessera-config: common_params_parse returned false; path was %s\n", path.c_str());
    }
    assert(ok);

    const common_tessera_params & tp = common_get_tessera_params();
    assert(tp.tessera_config_path == path);
    // config-supplied value (no CLI override):
    assert(tp.evolve_iters == 12);
    // CLI-overridden value (config said /tmp/from-config.npz, CLI said
    // /tmp/from-cli.npz):
    assert(tp.imatrix == "/tmp/from-cli.npz");

    // Cleanup
    unlink(path.c_str());
}

static void test_env_default(void) {
    printf("test-tessera-config: TESSERA_CONFIG env is honored when no CLI flag\n");

    const std::string ini =
        "[evolve]\n"
        "evolve-iters = 5\n";
    const std::string path = write_tmp(ini);

    setenv("TESSERA_CONFIG", path.c_str(), 1);

    auto argv = argv_holder({"binary", "-m", "/tmp/dummy.gguf"});
    common_params params;
    bool ok = common_params_parse((int) argv.ptrs.size(), argv.ptrs.data(), params, LLAMA_EXAMPLE_COMMON);
    assert(ok);

    const common_tessera_params & tp = common_get_tessera_params();
    assert(tp.tessera_config_path == path);
    assert(tp.evolve_iters == 5);

    unsetenv("TESSERA_CONFIG");
    unlink(path.c_str());
}

static void test_inline_kv(void) {
    printf("test-tessera-config: 'key:value' separator is accepted\n");

    // Tolerate the secondary ':' separator for users used to YAML-like
    // syntax. Both 'key = value' and 'key:value' must parse.
    const std::string ini =
        "[adapt]\n"
        "adapt-epsilon: 0.05\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    if (!ok) { fprintf(stderr, "%s", err.c_str()); assert(false); }
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    if (!ok) { fprintf(stderr, "%s", err.c_str()); assert(false); }
    assert(p.adapt_epsilon == 0.05);
}

static void test_runtime_sidecar(void) {
    printf("test-tessera-config: [runtime] sidecar keys accepted\n");

    // The dequant-dir/stride, matmul-output-dir/stride, and l15-dtype
    // keys write to module globals in tessera_debug / tessera_matmul_output.
    // We can only verify that parsing + apply do not fail (the value
    // is then accessible via those module getters in the running process).
    const std::string ini =
        "[runtime]\n"
        "dequant-dir = /tmp/dequant\n"
        "dequant-stride = 4\n"
        "matmul-output-dir = /tmp/matmul\n"
        "matmul-output-stride = 2\n"
        "[l15]\n"
        "l15-dtype = f32\n";
    tessera_config cfg;
    std::string err;
    bool ok = tessera_config_parse(ini, "<inline>", cfg, err);
    if (!ok) { fprintf(stderr, "%s", err.c_str()); assert(false); }
    common_tessera_params p;
    ok = tessera_config_apply(cfg, p, err);
    if (!ok) { fprintf(stderr, "%s", err.c_str()); assert(false); }
    assert(tessera_debug::l15_dtype() == "f32");
    assert(tessera_debug::dequant_stride() == 4);
    assert(tessera_matmul_output::matmul_output_stride() == 2);
    // restore defaults so other tests aren't affected
    tessera_debug::set_dequant_dir("");
    tessera_debug::set_dequant_stride(1);
    tessera_matmul_output::set_matmul_output_dir("");
    tessera_matmul_output::set_matmul_output_stride(1);
    tessera_debug::set_l15_dtype("f16");
}

int main(void) {
    test_parse_happy();
    test_parse_quoted();
    test_global_and_general();
    test_duplicate_key();
    test_unclosed_section();
    test_malformed_line();
    test_unknown_section();
    test_unknown_key();
    test_bad_enum();
    test_out_of_range();
    test_bad_int();
    test_load_file_missing();
    test_cli_overrides_config();
    test_env_default();
    test_inline_kv();
    test_runtime_sidecar();
    printf("test-tessera-config: all tests OK\n");
    return 0;
}
