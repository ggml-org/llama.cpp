// tests the model resolution of common/model-resolution on synthetic repo
// listings, then the end-to-end model handler assembly through the real CLI
// parsing, resolving offline against a fake on-disk model cache, no network
// access and no instrumentation of the tested code

#include "arg.h"
#include "common.h"
#include "hf-cache.h"
#include "log.h"
#include "model-resolution.h"
#include "speculative.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// the plan case being checked, named by the failure report below
static const char * current_case = "";

// independent of NDEBUG, so the checks stay alive in Release builds
#define REQUIRE(x) do {                                                        \
    if (!(x)) {                                                                \
        fprintf(stderr, "%s:%d: REQUIRE(%s) failed%s%s\n",                     \
                __FILE__, __LINE__, #x,                                        \
                *current_case ? " on case " : "", current_case);               \
        std::abort();                                                          \
    }                                                                          \
} while (0)

// fixtures mimicking real repo layouts

// flat layout in the style of ggml-org/gemma-4-31B-it-GGUF
static const std::vector<std::string> flat = {
    "README.md",
    "model-BF16.gguf",
    "model-Q4_K_M.gguf",
    "model-Q8_0.gguf",
    "mmproj-model-BF16.gguf",
    "mmproj-model-Q8_0.gguf",
    "mtp-model-BF16.gguf",
    "mtp-model-Q4_0.gguf",
    "mtp-model-Q8_0.gguf",
    "dflash-model-BF16.gguf",
    "dflash-model-Q8_0.gguf",
};

// quants in subdirectories with sharded files and root sidecars,
// in the style of stepfun-ai/Step-3.7-Flash-GGUF
static const std::vector<std::string> subdir = {
    "mmproj-model-f16.gguf",
    "model-mtp-BF16.gguf",
    "model-mtp-Q8_0.gguf",
    "Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
    "Q3_K_M/model-Q3_K_M-00002-of-00003.gguf",
    "Q3_K_M/model-Q3_K_M-00003-of-00003.gguf",
    "Q8_0/model-Q8_0-00001-of-00002.gguf",
    "Q8_0/model-Q8_0-00002-of-00002.gguf",
};

// sidecar quants exist where the full model quant does not,
// in the style of ggml-org/Qwen3.6-27B-GGUF
static const std::vector<std::string> hole = {
    "model-BF16.gguf",
    "model-Q4_K_M.gguf",
    "model-Q8_0.gguf",
    "mtp-model-BF16.gguf",
    "mtp-model-Q4_0.gguf",
    "mtp-model-Q8_0.gguf",
    "dflash-model-BF16.gguf",
    "dflash-model-Q8_0.gguf",
};

// unsloth-style naming with UD quants and a suffix MTP file
static const std::vector<std::string> unsloth = {
    "model-UD-Q8_K_XL.gguf",
    "mmproj-BF16.gguf",
    "model-MTP-BF16.gguf",
};

// bartowski-style vendor prefix and mradermacher-style dot quant
static const std::vector<std::string> vendors = {
    "TheDrummer_Model-24B-v4.1-Q8_0.gguf",
    "BlackSheep-24B.Q8_0.gguf",
};

// every speculative sidecar type at the same quant
static const std::vector<std::string> trio = {
    "model-Q8_0.gguf",
    "mtp-model-Q8_0.gguf",
    "dflash-model-Q8_0.gguf",
    "eagle3-model-Q8_0.gguf",
};

static const std::vector<std::string> dflash_only = {
    "model-Q8_0.gguf",
    "dflash-model-Q8_0.gguf",
};

static const std::vector<std::string> eagle3_only = {
    "model-Q8_0.gguf",
    "eagle3-model-Q8_0.gguf",
};

// a synthetic listing, the resolution only looks at the paths
static hf_cache::hf_files listing(const std::vector<std::string> & paths) {
    hf_cache::hf_files files;
    for (const auto & p : paths) {
        hf_cache::hf_file f;
        f.path = p;
        files.push_back(std::move(f));
    }
    return files;
}

// helpers of the model resolution unit

static void test_helpers() {
    printf("test-model-resolution: resolution helpers\n");

    // quant bits extraction on real world namings
    REQUIRE(model_resolution::extract_quant_bits("model-Q8_0.gguf")                 == 8);
    REQUIRE(model_resolution::extract_quant_bits("model-UD-Q2_K_XL.gguf")           == 2);
    REQUIRE(model_resolution::extract_quant_bits("model.i1-Q6_K.gguf")              == 6);
    REQUIRE(model_resolution::extract_quant_bits("model-BF16.gguf")                 == 16);
    REQUIRE(model_resolution::extract_quant_bits("model-MXFP4-00001-of-00002.gguf") == 4);

    // sidecar keywords are never a model, wherever they appear in the name
    REQUIRE(model_resolution::gguf_filename_is_model("model-Q8_0.gguf"));
    REQUIRE(!model_resolution::gguf_filename_is_model("mtp-model-Q8_0.gguf"));
    REQUIRE(!model_resolution::gguf_filename_is_model("model-mtp-Q8_0.gguf"));
    REQUIRE(!model_resolution::gguf_filename_is_model("dflash-model-Q8_0.gguf"));
    REQUIRE(!model_resolution::gguf_filename_is_model("eagle3-model-Q8_0.gguf"));
    REQUIRE(!model_resolution::gguf_filename_is_model("mmproj-model-Q8_0.gguf"));
    REQUIRE(!model_resolution::gguf_filename_is_model("model.txt"));

    // the sibling picker honors an exact tag over quant proximity
    auto files = listing({"mtp-model-BF16.gguf", "mtp-model-Q4_0.gguf", "mtp-model-Q8_0.gguf"});
    REQUIRE(model_resolution::find_best_sibling(files, "model-Q8_0.gguf", "mtp-", "Q4_0").path == "mtp-model-Q4_0.gguf");
    REQUIRE(model_resolution::find_best_sibling(files, "model-Q8_0.gguf", "mtp-").path        == "mtp-model-Q8_0.gguf");
}

// table-driven plan resolution, each case replayed on permutations of the
// listing to assert determinism, except the cases that legitimately depend
// on the listing order

struct plan_case {
    const char * name;
    const std::vector<std::string> & files;
    const char * tag;
    const char * hf_file;
    bool sidecars;        // request mmproj + mtp + dflash + eagle3
    bool order_dependent; // the expected pick depends on the listing order
    const char * primary;
    std::vector<std::string> model_files;
    const char * mmproj;
    const char * mtp;
    const char * dflash;
    const char * eagle3;
};

static const plan_case plan_cases[] = {
    // exact tag picks the matching primary, sidecars follow the tag
    {"flat exact tag", flat, "Q8_0", "", true, false,
     "model-Q8_0.gguf", {"model-Q8_0.gguf"},
     "mmproj-model-Q8_0.gguf", "mtp-model-Q8_0.gguf", "dflash-model-Q8_0.gguf", ""},

    // no tag falls back to the default quant preference
    {"flat default", flat, "", "", false, false,
     "model-Q4_K_M.gguf", {"model-Q4_K_M.gguf"},
     "", "", "", ""},

    // no tag and no default match falls back to the first model in the listing
    {"unsloth fallback", unsloth, "", "", true, true,
     "model-UD-Q8_K_XL.gguf", {"model-UD-Q8_K_XL.gguf"},
     "mmproj-BF16.gguf", "", "", ""},

    // explicit hf_file picks that exact file
    {"flat hf_file", flat, "", "model-BF16.gguf", false, false,
     "model-BF16.gguf", {"model-BF16.gguf"},
     "", "", "", ""},

    // missing hf_file resolves nothing
    {"flat missing hf_file", flat, "", "nope.gguf", false, false,
     "", {},
     "", "", "", ""},

    // a sharded primary brings all its parts, a subdir primary finds the root sidecar
    {"subdir shards", subdir, "Q3_K_M", "", true, false,
     "Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
     {"Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
      "Q3_K_M/model-Q3_K_M-00002-of-00003.gguf",
      "Q3_K_M/model-Q3_K_M-00003-of-00003.gguf"},
     "mmproj-model-f16.gguf", "model-mtp-Q8_0.gguf", "", ""},

    // a tag with no matching full model still resolves the requested sidecars
    {"hole tag sidecar", hole, "Q4_0", "", true, false,
     "", {},
     "", "mtp-model-Q4_0.gguf", "dflash-model-Q8_0.gguf", ""},

    // the same tag without a requested sidecar resolves nothing
    {"hole tag alone", hole, "Q4_0", "", false, false,
     "", {},
     "", "", "", ""},

    // no tag anchors the sidecars on the primary quant
    {"hole default anchor", hole, "", "", true, false,
     "model-Q4_K_M.gguf", {"model-Q4_K_M.gguf"},
     "", "mtp-model-Q4_0.gguf", "dflash-model-Q8_0.gguf", ""},

    // the mtp- keyword is case sensitive, a suffix -MTP file is not discovered
    {"unsloth suffix mtp", unsloth, "Q8_K_XL", "", true, false,
     "model-UD-Q8_K_XL.gguf", {"model-UD-Q8_K_XL.gguf"},
     "mmproj-BF16.gguf", "", "", ""},

    // vendor prefixes and the dot quant convention both match the tag,
    // first match wins between two files at the same quant
    {"vendor prefix", vendors, "Q8_0", "", false, true,
     "TheDrummer_Model-24B-v4.1-Q8_0.gguf", {"TheDrummer_Model-24B-v4.1-Q8_0.gguf"},
     "", "", "", ""},

    // every sidecar type resolves at the tag
    {"trio exact tag", trio, "Q8_0", "", true, false,
     "model-Q8_0.gguf", {"model-Q8_0.gguf"},
     "", "mtp-model-Q8_0.gguf", "dflash-model-Q8_0.gguf", "eagle3-model-Q8_0.gguf"},
};

static void check_plan(const plan_case & c, const hf_cache::hf_files & files) {
    current_case = c.name;
    model_resolution::opts opts;
    opts.mmproj = c.sidecars;
    opts.mtp    = c.sidecars;
    opts.dflash = c.sidecars;
    opts.eagle3 = c.sidecars;

    auto plan = model_resolution::resolve(files, "test/repo", c.tag, c.hf_file, opts);

    REQUIRE(plan.primary.path == c.primary);
    REQUIRE(plan.mmproj.path  == c.mmproj);
    REQUIRE(plan.mtp.path     == c.mtp);
    REQUIRE(plan.dflash.path  == c.dflash);
    REQUIRE(plan.eagle3.path  == c.eagle3);

    // the exact shard set, order insensitive, with the primary as first split
    std::vector<std::string> actual;
    for (const auto & f : plan.model_files) {
        actual.push_back(f.path);
    }
    std::sort(actual.begin(), actual.end());
    auto expected = c.model_files;
    std::sort(expected.begin(), expected.end());
    REQUIRE(actual == expected);
    if (!expected.empty()) {
        REQUIRE(plan.primary.path == expected.front());
    }

    // invariant: the primary is never a sidecar file
    if (!plan.primary.path.empty()) {
        REQUIRE(model_resolution::gguf_filename_is_model(plan.primary.path));
    }
    current_case = "";
}

static void test_plan_resolution() {
    printf("test-model-resolution: plan resolution on %zu cases\n", sizeof(plan_cases) / sizeof(plan_cases[0]));

    for (const auto & c : plan_cases) {
        auto files = listing(c.files);

        // invariant: the resolution is insensitive to the listing order
        for (size_t rot = 0; rot < files.size(); ++rot) {
            if (c.order_dependent && rot > 0) {
                continue;
            }
            auto permuted = files;
            std::rotate(permuted.begin(), permuted.begin() + rot, permuted.end());
            if (rot % 2 == 1) {
                std::reverse(permuted.begin(), permuted.end());
            }
            check_plan(c, permuted);
        }
    }
}

// end-to-end assembly: real CLI parsing and real handler init, --offline
// resolves against the fake on-disk cache below and skips the downloads
// while on_done still wires the params

// the cache layout expected by hf-cache.cpp, refs/main names a commit hash
// of 40 hex characters and snapshots/<commit> holds the files of that commit
static const fs::path cache_dir = fs::temp_directory_path() / "test-model-resolution-cache";
static const std::string commit = std::string(40, 'c');

static fs::path repo_dir(std::string repo_id) {
    string_replace_all(repo_id, "/", "--");
    return cache_dir / ("models--" + repo_id);
}

static void write_repo(const std::string & repo_id, const std::vector<std::string> & paths) {
    fs::create_directories(repo_dir(repo_id) / "refs");
    std::ofstream(repo_dir(repo_id) / "refs" / "main") << commit;
    for (const auto & p : paths) {
        fs::path file = repo_dir(repo_id) / "snapshots" / commit / fs::path(p);
        fs::create_directories(file.parent_path());
        std::ofstream(file) << "";
    }
}

// the local path the handler is expected to wire for a cached file
static std::string cached(const std::string & repo_id, const std::string & path) {
    return (repo_dir(repo_id) / "snapshots" / commit / fs::path(path)).string();
}

static void assemble(std::vector<std::string> argv, common_params & params) {
    argv.push_back("--offline");
    std::vector<char *> cargv;
    for (auto & a : argv) {
        cargv.push_back(a.data());
    }
    REQUIRE(common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_SERVER));

    auto handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
    common_models_handler_apply(handler, params);
}

static void test_task_assembly() {
    printf("test-model-resolution: end-to-end assembly\n");

    write_repo("test/main",   flat);
    write_repo("test/hole",   hole);
    write_repo("test/trio",   trio);
    write_repo("test/dflash", dflash_only);
    write_repo("test/eagle3", eagle3_only);
    write_repo("test/small",  {"draft-model-Q4_K_M.gguf"});
    write_repo("test/preset", {"preset.ini", "model-Q8_0.gguf"});

    {
        // plain -hf wires the model and its mmproj, nothing speculative
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0"}, params);
        REQUIRE(params.model.path  == cached("test/main", "model-Q8_0.gguf"));
        REQUIRE(params.mmproj.path == cached("test/main", "mmproj-model-Q8_0.gguf"));
        REQUIRE(params.speculative.draft.mparams.path.empty());
    }
    {
        // --no-mmproj disables the mmproj discovery
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--no-mmproj"}, params);
        REQUIRE(params.mmproj.path.empty());
    }
    {
        // an explicit --mmproj wins over the discovery
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--mmproj", "/local/mmproj.gguf"}, params);
        REQUIRE(params.mmproj.path == "/local/mmproj.gguf");
    }
    {
        // -hf with a spec type wires the sidecar of the main repo as fallback draft
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--spec-type", "draft-mtp"}, params);
        REQUIRE(params.speculative.draft.mparams.path == cached("test/main", "mtp-model-Q8_0.gguf"));
    }
    {
        // -hfd with a spec type wires the draft repo sidecar at its tag,
        // not its full model, and suppresses the main repo fallback
        common_params params;
        assemble({"server", "-hf", "test/hole:Q8_0", "-hfd", "test/hole:Q4_0", "--spec-type", "draft-mtp"}, params);
        REQUIRE(params.speculative.draft.mparams.path == cached("test/hole", "mtp-model-Q4_0.gguf"));
    }
    {
        // an explicit -md file wins over the sidecar resolution
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/main", "-md", "mtp-model-BF16.gguf", "--spec-type", "draft-mtp"}, params);
        REQUIRE(params.speculative.draft.mparams.path == cached("test/main", "mtp-model-BF16.gguf"));
    }
    {
        // -hfd without a spec type auto-selects the type, mtp first when all ship
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/trio:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP});
        REQUIRE(params.speculative.draft.mparams.path == cached("test/trio", "mtp-model-Q8_0.gguf"));
    }
    {
        // auto-selection with only a dflash sidecar
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/dflash:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH});
        REQUIRE(params.speculative.draft.mparams.path == cached("test/dflash", "dflash-model-Q8_0.gguf"));
    }
    {
        // auto-selection with only an eagle3 sidecar
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/eagle3:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3});
        REQUIRE(params.speculative.draft.mparams.path == cached("test/eagle3", "eagle3-model-Q8_0.gguf"));
    }
    {
        // -hfd on a repo without sidecars keeps resolving a full model as draft
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/small"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE});
        REQUIRE(params.speculative.draft.mparams.path == cached("test/small", "draft-model-Q4_K_M.gguf"));
    }
    {
        // a preset repo wires the preset and clears the model for router mode
        common_params params;
        assemble({"server", "-hf", "test/preset"}, params);
        REQUIRE(params.models_preset == cached("test/preset", "preset.ini"));
        REQUIRE(params.model.path.empty());
        REQUIRE(params.model.hf_repo.empty());
    }
}

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main(void) {
    // the negative cases legitimately log errors on every permutation,
    // keep the output down to the printf reports and the failure reports
    common_log_pause(common_log_main());

    // the cache location is read once by hf-cache.cpp, point it at the fake
    // cache before anything else touches it
    set_env("LLAMA_CACHE", cache_dir.string().c_str());
    fs::remove_all(cache_dir);

    test_helpers();
    test_plan_resolution();
    test_task_assembly();

    fs::remove_all(cache_dir);
    printf("test-model-resolution: all tests OK\n");
    return 0;
}
