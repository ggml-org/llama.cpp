// tests the HF model resolution and the model handler assembly end-to-end on
// synthetic repo listings: a local httplib server bound to the loopback
// serves hardcoded HF API responses, so the real client, hf_cache, resolution
// and CLI parsing run against them without external network access

#include "arg.h"
#include "common.h"
#include "download.h"
#include "http.h"
#include "log.h"

#include "json.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>
#include <string>
#include <vector>

// the case and reordering being checked, printed with every failure
static std::string g_context;

// independent of NDEBUG, so the checks stay alive in Release builds
#define REQUIRE(x) do {                                                         \
    if (!(x)) {                                                                 \
        fprintf(stderr, "%s:%d: [%s] REQUIRE(%s) failed\n",                     \
                __FILE__, __LINE__, g_context.c_str(), #x);                     \
        std::abort();                                                           \
    }                                                                           \
} while (0)

#define REQUIRE_EQ(actual, expected) do {                                       \
    if (!((actual) == (expected))) {                                            \
        fprintf(stderr, "%s:%d: [%s] REQUIRE_EQ(%s, %s) failed\n  actual:   '%s'\n  expected: '%s'\n", \
                __FILE__, __LINE__, g_context.c_str(), #actual, #expected,      \
                std::string(actual).c_str(), std::string(expected).c_str());    \
        std::abort();                                                           \
    }                                                                           \
} while (0)

//
// synthetic repos keyed by repo id, served over the loopback by a real
// httplib server, so the tested code runs its own client and transport
//

static std::map<std::string, std::vector<std::string>> g_repos;

static const char * COMMIT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

// the server lives in main, so its destructor runs before the static teardown
// tears down the winsock state httplib brings in
static void serve_repos(httplib::Server & server) {
    server.Get(R"(/api/models/(.+)/refs)", [](const httplib::Request & req, httplib::Response & res) {
        if (g_repos.count(req.matches[1])) {
            res.set_content(common_json{{"branches", common_json::array({ common_json{{"name", "main"}, {"targetCommit", COMMIT}} })}}.dump(),
                            "application/json");
        } else {
            res.status = 404;
        }
    });
    server.Get(R"(/api/models/(.+)/tree/.+)", [](const httplib::Request & req, httplib::Response & res) {
        if (!g_repos.count(req.matches[1])) {
            res.status = 404;
            return;
        }
        auto files = common_json::array();
        size_t i = 0;
        for (const auto & p : g_repos[req.matches[1]]) {
            char oid[41];
            snprintf(oid, sizeof(oid), "%040lx", (unsigned long) ++i);
            files.push_back({{"type", "file"}, {"path", p}, {"size", 1}, {"oid", oid}});
        }
        res.set_content(files.dump(), "application/json");
    });
}

static common_params_model model_ref(const std::string & hf_repo, const std::string & hf_file = "") {
    common_params_model m;
    m.hf_repo = hf_repo;
    m.hf_file = hf_file;
    return m;
}

// the model cache is isolated under a temporary directory named after the
// loopback port, so concurrent runs on a shared machine keep their own, and
// the local path the handler wires for a file is snapshots/<commit>/<path>
static std::filesystem::path cache_dir;

static std::string cached(std::string repo_id, const std::string & path) {
    string_replace_all(repo_id, "/", "--");
    return (cache_dir / ("models--" + repo_id) / "snapshots" / COMMIT / path).string();
}

//
// fixtures mimicking real repo layouts
//

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
static const std::vector<std::string> quad = {
    "model-Q8_0.gguf",
    "mtp-model-Q8_0.gguf",
    "dflash-model-Q8_0.gguf",
    "eagle3-model-Q8_0.gguf",
    "dspark-model-Q8_0.gguf",
};

static const std::vector<std::string> dflash_only = {
    "model-Q8_0.gguf",
    "dflash-model-Q8_0.gguf",
};

static const std::vector<std::string> eagle3_only = {
    "model-Q8_0.gguf",
    "eagle3-model-Q8_0.gguf",
};

// a single full quant with dspark sidecars at other quants,
// in the style of ggml-org/DeepSeek-V4-Flash-0731-GGUF
static const std::vector<std::string> spark = {
    "README.md",
    "model-MXFP4.gguf",
    "dspark-model-BF16.gguf",
    "dspark-model-MXFP4.gguf",
};

// dspark outranks dflash in the type auto-selection
static const std::vector<std::string> dspark_dflash = {
    "model-Q8_0.gguf",
    "dflash-model-Q8_0.gguf",
    "dspark-model-Q8_0.gguf",
};

//
// table-driven plan resolution through the real entry point,
// each case replayed on multiple deterministic reorderings of the listing,
// except the cases whose pick legitimately depends on the listing order
//

struct plan_case {
    const char * name;
    const std::vector<std::string> files;
    const char * hf_repo;
    const char * hf_file;
    bool sidecars;        // request mmproj + mtp + dflash + eagle3 + dspark
    bool order_dependent; // the expected pick depends on the listing order
    const char * primary;
    std::vector<std::string> model_files;
    const char * mmproj;
    const char * mtp;
    const char * dflash;
    const char * eagle3;
    const char * dspark;
};

static const plan_case plan_cases[] = {
    // exact tag picks the matching primary, sidecars follow the tag
    {"flat exact tag", flat, "test/repo:Q8_0", "", true, false,
     "model-Q8_0.gguf", {"model-Q8_0.gguf"},
     "mmproj-model-Q8_0.gguf", "mtp-model-Q8_0.gguf", "dflash-model-Q8_0.gguf", "", ""},

    // no tag falls back to the default quant preference
    {"flat default", flat, "test/repo", "", false, false,
     "model-Q4_K_M.gguf", {"model-Q4_K_M.gguf"},
     "", "", "", "", ""},

    // no tag and no default match falls back to the first model in the listing
    {"unsloth fallback", unsloth, "test/repo", "", true, true,
     "model-UD-Q8_K_XL.gguf", {"model-UD-Q8_K_XL.gguf"},
     "mmproj-BF16.gguf", "", "", "", ""},

    // explicit hf_file picks that exact file
    {"flat hf_file", flat, "test/repo", "model-BF16.gguf", false, false,
     "model-BF16.gguf", {"model-BF16.gguf"},
     "", "", "", "", ""},

    // missing hf_file resolves nothing
    {"flat missing hf_file", flat, "test/repo", "nope.gguf", false, false,
     "", {},
     "", "", "", "", ""},

    // a sharded primary brings all its parts, a subdir primary finds the root sidecar
    {"subdir shards", subdir, "test/repo:Q3_K_M", "", true, false,
     "Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
     {"Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
      "Q3_K_M/model-Q3_K_M-00002-of-00003.gguf",
      "Q3_K_M/model-Q3_K_M-00003-of-00003.gguf"},
     "mmproj-model-f16.gguf", "model-mtp-Q8_0.gguf", "", "", ""},

    // a tag with no matching full model still resolves the requested sidecars
    {"hole tag sidecar", hole, "test/repo:Q4_0", "", true, false,
     "", {},
     "", "mtp-model-Q4_0.gguf", "dflash-model-Q8_0.gguf", "", ""},

    // the same tag without a requested sidecar resolves nothing
    {"hole tag alone", hole, "test/repo:Q4_0", "", false, false,
     "", {},
     "", "", "", "", ""},

    // no tag anchors the sidecars on the primary quant
    {"hole default anchor", hole, "test/repo", "", true, false,
     "model-Q4_K_M.gguf", {"model-Q4_K_M.gguf"},
     "", "mtp-model-Q4_0.gguf", "dflash-model-Q8_0.gguf", "", ""},

    // the mtp- keyword is case sensitive, a suffix -MTP file is not discovered
    {"unsloth suffix mtp", unsloth, "test/repo:Q8_K_XL", "", true, false,
     "model-UD-Q8_K_XL.gguf", {"model-UD-Q8_K_XL.gguf"},
     "mmproj-BF16.gguf", "", "", "", ""},

    // vendor prefixes and the dot quant convention both match the tag,
    // first match wins between two files at the same quant
    {"vendor prefix", vendors, "test/repo:Q8_0", "", false, true,
     "TheDrummer_Model-24B-v4.1-Q8_0.gguf", {"TheDrummer_Model-24B-v4.1-Q8_0.gguf"},
     "", "", "", "", ""},

    // every sidecar type resolves at the tag
    {"quad exact tag", quad, "test/repo:Q8_0", "", true, false,
     "model-Q8_0.gguf", {"model-Q8_0.gguf"},
     "", "mtp-model-Q8_0.gguf", "dflash-model-Q8_0.gguf", "eagle3-model-Q8_0.gguf", "dspark-model-Q8_0.gguf"},

    // no tag anchors the dspark sidecar on the only full quant
    {"spark default anchor", spark, "test/repo", "", true, false,
     "model-MXFP4.gguf", {"model-MXFP4.gguf"},
     "", "", "", "", "dspark-model-MXFP4.gguf"},

    // a tag with no matching full model still resolves the exact dspark sidecar
    {"spark tag sidecar", spark, "test/repo:BF16", "", true, false,
     "", {},
     "", "", "", "", "dspark-model-BF16.gguf"},
};

static void check_plan(const plan_case & c) {
    common_download_opts opts;
    opts.download_mmproj = c.sidecars;
    opts.download_mtp    = c.sidecars;
    opts.download_dflash = c.sidecars;
    opts.download_eagle3 = c.sidecars;
    opts.download_dspark = c.sidecars;

    auto plan = common_download_get_hf_plan(model_ref(c.hf_repo, c.hf_file), opts);

    REQUIRE_EQ(plan.primary.path, c.primary);
    REQUIRE_EQ(plan.mmproj.path,  c.mmproj);
    REQUIRE_EQ(plan.mtp.path,     c.mtp);
    REQUIRE_EQ(plan.dflash.path,  c.dflash);
    REQUIRE_EQ(plan.eagle3.path,  c.eagle3);
    REQUIRE_EQ(plan.dspark.path,  c.dspark);

    // exact shard set, order insensitive; the primary must be the first split
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
}

static void test_plan_resolution() {
    printf("test-model-resolution: plan resolution on %zu cases\n", sizeof(plan_cases) / sizeof(plan_cases[0]));

    for (const auto & c : plan_cases) {
        printf("  %s\n", c.name);
        // invariant: the resolution is insensitive to the listing order
        for (size_t rot = 0; rot < c.files.size(); ++rot) {
            if (c.order_dependent && rot > 0) {
                continue;
            }
            g_context = std::string(c.name) + ", reordering " + std::to_string(rot);
            auto files = c.files;
            std::rotate(files.begin(), files.begin() + rot, files.end());
            if (rot % 2 == 1) {
                std::reverse(files.begin(), files.end());
            }
            g_repos["test/repo"] = files;
            check_plan(c);
        }
    }
    g_repos.clear();
}

// put a file in the cache the way a finished download leaves it, refs/main included
static void cache_put(const std::string & repo_id, const std::string & path) {
    namespace fs = std::filesystem;

    const fs::path local = cached(repo_id, path);
    fs::create_directories(local.parent_path());
    { std::ofstream(local) << "gguf"; }

    const fs::path ref = local.parent_path().parent_path().parent_path() / "refs" / "main";
    fs::create_directories(ref.parent_path());
    if (!fs::exists(ref)) {
        std::ofstream(ref) << COMMIT << "\n";
    }
}

//
// partials of a download: the files of its plan not in the cache yet, next to
// the blobs, named after the oids the listing serves
//

static void test_plan_partials() {
    namespace fs = std::filesystem;

    printf("test-model-resolution: plan partials\n");

    g_context = "plan partials";
    g_repos.clear();
    g_repos["test/partial"] = { "model-Q4_0.gguf", "mmproj-model-F16.gguf", "mtp-model-Q4_0.gguf" };

    // the mmproj is complete, the model and its mtp are still to download
    cache_put("test/partial", "mmproj-model-F16.gguf");

    common_download_opts opts;
    opts.download_mmproj = true;
    opts.download_mtp    = true;

    const auto parts = common_download_plan_partials(common_download_get_hf_plan(model_ref("test/partial:Q4_0"), opts));

    const fs::path blobs = cache_dir / "models--test--partial" / "blobs";
    REQUIRE(parts.size() == 2);
    REQUIRE(parts[0].kind.empty());
    REQUIRE(fs::path(parts[0].path).lexically_normal() == (blobs / "0000000000000000000000000000000000000001.downloadInProgress").lexically_normal());
    REQUIRE_EQ(parts[1].kind, "mtp");
    REQUIRE(fs::path(parts[1].path).lexically_normal() == (blobs / "0000000000000000000000000000000000000003.downloadInProgress").lexically_normal());

    fs::remove_all(cache_dir / "models--test--partial");
    g_repos.clear();
}

//
// removal of a sidecar alone: its snapshot entry goes, the model keeps its files
//

static void test_sidecar_removal() {
    namespace fs = std::filesystem;

    printf("test-model-resolution: sidecar removal\n");

    g_context = "sidecar removal";
    g_repos.clear();

    cache_put("test/sidecar", "model-Q4_0.gguf");
    cache_put("test/sidecar", "dflash-model-Q4_0.gguf");
    cache_put("test/sidecar", "mtp-model-Q4_0.gguf");

    common_download_opts opts;
    opts.download_dflash = true;

    // the selected sidecar goes, the other one and the model stay
    REQUIRE(common_download_remove_sidecars("test/sidecar:Q4_0", opts));
    REQUIRE(!fs::exists(cached("test/sidecar", "dflash-model-Q4_0.gguf")));
    REQUIRE(fs::exists(cached("test/sidecar", "mtp-model-Q4_0.gguf")));
    REQUIRE(fs::exists(cached("test/sidecar", "model-Q4_0.gguf")));
    REQUIRE(!common_download_remove_sidecars("test/sidecar:Q4_0", opts));

    fs::remove_all(cache_dir / "models--test--sidecar");
}

//
// removal of a tag keeps the files another cached tag still loads: a sidecar
// named after the removed tag that another quant resolves stays on disk
//

static void test_shared_removal() {
    namespace fs = std::filesystem;

    printf("test-model-resolution: shared removal\n");

    g_context = "shared removal";
    g_repos.clear();

    for (const char * f : { "model-Q8_0.gguf", "model-BF16.gguf", "mmproj-model-BF16.gguf", "mtp-model-BF16.gguf" }) {
        cache_put("test/shared", f);
    }

    // the BF16 model goes, the mmproj and the mtp Q8_0 resolves stay
    REQUIRE(common_download_remove("test/shared:BF16"));
    REQUIRE(!fs::exists(cached("test/shared", "model-BF16.gguf")));
    REQUIRE(fs::exists(cached("test/shared", "mmproj-model-BF16.gguf")));
    REQUIRE(fs::exists(cached("test/shared", "mtp-model-BF16.gguf")));
    REQUIRE(fs::exists(cached("test/shared", "model-Q8_0.gguf")));

    fs::remove_all(cache_dir / "models--test--shared");

    // with no other tag, the files of the removed tag all go, sidecars included
    for (const char * f : { "model-BF16.gguf", "mmproj-model-BF16.gguf", "mtp-model-BF16.gguf" }) {
        cache_put("test/alone", f);
    }

    REQUIRE(common_download_remove("test/alone:BF16"));
    REQUIRE(!fs::exists(cached("test/alone", "model-BF16.gguf")));
    REQUIRE(!fs::exists(cached("test/alone", "mmproj-model-BF16.gguf")));
    REQUIRE(!fs::exists(cached("test/alone", "mtp-model-BF16.gguf")));
    REQUIRE(!common_download_remove("test/alone:BF16"));

    fs::remove_all(cache_dir / "models--test--alone");
}

//
// real repo listings captured from the HF API on 2026-10-14: 1221 GGUF repos
// of the major publishers reduced to one listing per naming pattern, each
// pruned to whole quant groups; the expectations are what each file really is
// per its GGUF header, so a mismatch is a resolution bug, not a fixture choice
//

// a file name, or with `parts` a shard prefix standing for the verbatim names
// <prefix>-00001-of-0000n.gguf .. <prefix>-0000n-of-0000n.gguf
struct real_file {
    const char * name;
    int          parts = 0;
};

// a nullptr field is not asserted, where the right pick is not settled
struct real_plan {
    const char * tag;     // the request is <repo>:<tag>, or <repo> when empty
    const char * primary;
    int          parts;   // number of model files, all splits of the primary
    const char * mmproj;
    const char * mtp;
    const char * dflash;
    const char * eagle3;
    const char * dspark;
};

struct real_repo {
    const char * id;
    std::vector<real_file> files;
    std::vector<real_plan> plans;
    std::vector<std::string> listing; // the tags the cache listing reports
};

static const real_repo real_repos[] = {
    {"a-mo-yehia/Qwen3-VL-8B-Contracts-OCR-GGUF",
     {{"gguf/mmproj_f16.gguf"},
      {"gguf/model_f16.gguf"},
      {"gguf/model_q4_k_m.gguf"},
      {"gguf/model_q8_0.gguf"}},
     {
      {"", "gguf/model_q4_k_m.gguf", 1, "gguf/mmproj_f16.gguf", "", "", "", ""},
      {"F16", "gguf/model_f16.gguf", 1, "gguf/mmproj_f16.gguf", "", "", "", ""},
      {"Q4_K_M", "gguf/model_q4_k_m.gguf", 1, "gguf/mmproj_f16.gguf", "", "", "", ""},
      {"Q8_0", "gguf/model_q8_0.gguf", 1, "gguf/mmproj_f16.gguf", "", "", "", ""}},
     {"F16", "Q4_K_M", "Q8_0"}},

    {"AesSedai/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF",
     {{"Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M", 3},
      {"imatrix-2.gguf"},
      {"imatrix.gguf"}},
     {
      {"", "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf", 3, "", "", "", "", ""},
      {"Q4_K_M", "Q4_K_M/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-Q4_K_M-00001-of-00003.gguf", 3, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"bartowski/Qwen3.8-Flash-Next-GGUF",
     {{"Qwen3.8-Flash-Next-Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M", 4},
      {"Qwen3.8-Flash-Next-imatrix.gguf"},
      {"mmproj-Qwen3.8-Flash-Next-bf16.gguf"},
      {"mmproj-Qwen3.8-Flash-Next-f16.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q4_0.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q8_0.gguf"},
      {"mtp-Qwen3.8-Flash-Next-bf16.gguf"}},
     {
      {"", "Qwen3.8-Flash-Next-Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf", 4, "mmproj-Qwen3.8-Flash-Next-bf16.gguf", "mtp-Qwen3.8-Flash-Next-Q4_0.gguf", "", "", ""},
      {"Q4_K_M", "Qwen3.8-Flash-Next-Q4_K_M/Qwen3.8-Flash-Next-Q4_K_M-00001-of-00004.gguf", 4, "mmproj-Qwen3.8-Flash-Next-bf16.gguf", "mtp-Qwen3.8-Flash-Next-Q4_0.gguf", "", "", ""}},
     {"Q4_K_M"}},

    {"DevQuasar/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp-GGUF",
     {{"MXFP4_MOE/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.MXFP4_MOE.gguf", 12},
      {"Q2_K/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.Q2_K.gguf", 8},
      {"Q4_K_M/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.Q4_K_M.gguf", 14},
      {"mmproj-deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf"}},
     {
      {"", "Q4_K_M/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.Q4_K_M.gguf-00001-of-00014.gguf", 14, "mmproj-deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf", "", "", "", ""},
      {"MXFP4_MOE", "MXFP4_MOE/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.MXFP4_MOE.gguf-00001-of-00012.gguf", 12, "mmproj-deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf", "", "", "", ""},
      {"Q2_K", "Q2_K/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.Q2_K.gguf-00001-of-00008.gguf", 8, "mmproj-deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf", "", "", "", ""},
      {"Q4_K_M", "Q4_K_M/deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf.Q4_K_M.gguf-00001-of-00014.gguf", 14, "mmproj-deepseek-ai.DeepSeek-V4-Flash-Vision-Exp.f16.gguf", "", "", "", ""}},
     {"MXFP4_MOE", "Q2_K", "Q4_K_M"}},

    {"DevQuasar/meta-models.Muse-Glimmer-30B-GGUF",
     {{"Q4_K_M/meta-models.Muse-Glimmer-30B.f16.gguf.Q4_K_M.gguf"},
      {"dflash-meta-models.Muse-Glimmer-30B.Q4_K_M.gguf"},
      {"dflash-meta-models.Muse-Glimmer-30B.Q8_0.gguf"},
      {"dflash-meta-models.Muse-Glimmer-30B.bf16.gguf"},
      {"mmproj-meta-models.Muse-Glimmer-30B.f16.gguf"}},
     {
      {"", "Q4_K_M/meta-models.Muse-Glimmer-30B.f16.gguf.Q4_K_M.gguf", 1, "mmproj-meta-models.Muse-Glimmer-30B.f16.gguf", "", "dflash-meta-models.Muse-Glimmer-30B.Q4_K_M.gguf", "", ""},
      {"Q4_K_M", "Q4_K_M/meta-models.Muse-Glimmer-30B.f16.gguf.Q4_K_M.gguf", 1, "mmproj-meta-models.Muse-Glimmer-30B.f16.gguf", "", "dflash-meta-models.Muse-Glimmer-30B.Q4_K_M.gguf", "", ""}},
     {"Q4_K_M"}},

    {"DevQuasar/microsoft.Phi-4-reasoning-vision-15B-GGUF",
     {{"microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q2_K.gguf"},
      {"microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q3_K_M.gguf"},
      {"microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q4_K_M.gguf"},
      {"microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q5_K_M.gguf"},
      {"microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q6_K.gguf"},
      {"microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q8_0.gguf"},
      {"mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf"}},
     {
      {"", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q4_K_M.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""},
      {"Q2_K", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q2_K.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""},
      {"Q3_K_M", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q3_K_M.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""},
      {"Q4_K_M", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q4_K_M.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""},
      {"Q5_K_M", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q5_K_M.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""},
      {"Q6_K", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q6_K.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""},
      {"Q8_0", "microsoft.Phi-4-reasoning-vision-15B.f16.gguf.Q8_0.gguf", 1, "mmproj-microsoft.Phi-4-reasoning-vision-15B.f16.gguf", "", "", "", ""}},
     {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}},

    {"DevQuasar/moonshotai.Kimi-K3-GGUF",
     {{"F16/moonshotai.Kimi-K3.bf16", 100},
      {"Q2_K/moonshotai.Kimi-K3.Q2_K", 70}},
     {
      {"", "F16/moonshotai.Kimi-K3.bf16-00001-of-00100.gguf", 100, "", "", "", "", ""},
      {"BF16", "F16/moonshotai.Kimi-K3.bf16-00001-of-00100.gguf", 100, "", "", "", "", ""},
      {"Q2_K", "Q2_K/moonshotai.Kimi-K3.Q2_K-00001-of-00070.gguf", 70, "", "", "", "", ""}},
     {"BF16", "Q2_K"}},

    {"DevQuasar/NOT_WOKRKING-LGAI-EXAONE.K-EXAONE-2.0-750B-A37B-GGUF",
     {{"Q2_K/LGAI-EXAONE.K-EXAONE-2.0-750B-A37B.Q2_K", 18}},
     {
      {"", "Q2_K/LGAI-EXAONE.K-EXAONE-2.0-750B-A37B.Q2_K-00001-of-00018.gguf", 18, "", "", "", "", ""},
      {"Q2_K", "Q2_K/LGAI-EXAONE.K-EXAONE-2.0-750B-A37B.Q2_K-00001-of-00018.gguf", 18, "", "", "", "", ""}},
     {"Q2_K"}},

    {"DevQuasar/nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-GGUF",
     {{"Q4_K_M/nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.f16.gguf.Q4_K_M.gguf"},
      {"mtp-nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.f16.gguf"}},
     {
      {"", "Q4_K_M/nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.f16.gguf.Q4_K_M.gguf", 1, "", "mtp-nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.f16.gguf", "", "", ""},
      {"Q4_K_M", "Q4_K_M/nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.f16.gguf.Q4_K_M.gguf", 1, "", "mtp-nvidia.NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.f16.gguf", "", "", ""}},
     {"Q4_K_M"}},

    {"DevQuasar/poolside.Laguna-S-2.1-GGUF",
     {{"Q4_K_M/poolside.Laguna-S-2.1.f16.gguf.Q4_K_M.gguf", 6}},
     {
      {"", "Q4_K_M/poolside.Laguna-S-2.1.f16.gguf.Q4_K_M.gguf-00001-of-00006.gguf", 6, "", "", "", "", ""},
      {"Q4_K_M", "Q4_K_M/poolside.Laguna-S-2.1.f16.gguf.Q4_K_M.gguf-00001-of-00006.gguf", 6, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"DevQuasar/Qwen.Qwen3-Coder-30B-A3B-Instruct-GGUF",
     {{"Qwen.Qwen3-Coder-30B-A3B-Instruct.Q4_K_M.gguf"},
      {"Qwen.Qwen3-Coder-30B-A3B-Instruct.f16", 2}},
     {
      {"", "Qwen.Qwen3-Coder-30B-A3B-Instruct.Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen.Qwen3-Coder-30B-A3B-Instruct.Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"F16", "Qwen.Qwen3-Coder-30B-A3B-Instruct.f16-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"F16", "Q4_K_M"}},

    {"DevQuasar/sarvamai.sarvam-30b-GGUF",
     {{"sarvamai.sarvam-30b.f16.gguf.Q4_K_M.gguf"}},
     {
      {"", "sarvamai.sarvam-30b.f16.gguf.Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "sarvamai.sarvam-30b.f16.gguf.Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"DevQuasar/xai-org.grok-2-GGUF",
     {{"imatrix/xai-org.grok_imatrix.gguf"},
      {"xai-org.grok-2.Q4_K_M", 12}},
     {
      {"", "xai-org.grok-2.Q4_K_M-00001-of-00012.gguf", 12, "", "", "", "", ""},
      {"Q4_K_M", "xai-org.grok-2.Q4_K_M-00001-of-00012.gguf", 12, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"ggml-org/DeepSeek-V4-Flash-Vision-Exp-GGUF",
     {{"DeepSeek-V4-Flash-Vision-Exp-MXFP4", 2},
      {"DeepSeek-V4-Flash-Vision-Exp-Q2_K", 2},
      {"DeepSeek-V4-Flash-Vision-Exp-Q2_K_S", 2},
      {"dspark-DeepSeek-V4-Flash-Vision-Exp-BF16.gguf"},
      {"dspark-DeepSeek-V4-Flash-Vision-Exp-MXFP4.gguf"},
      {"mmproj-DeepSeek-V4-Flash-Vision-Exp-BF16.gguf"},
      {"mmproj-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf"}},
     {
      {"", "DeepSeek-V4-Flash-Vision-Exp-MXFP4-00001-of-00002.gguf", 2, "mmproj-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf", "", "", "", "dspark-DeepSeek-V4-Flash-Vision-Exp-MXFP4.gguf"},
      {"MXFP4", "DeepSeek-V4-Flash-Vision-Exp-MXFP4-00001-of-00002.gguf", 2, "mmproj-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf", "", "", "", "dspark-DeepSeek-V4-Flash-Vision-Exp-MXFP4.gguf"},
      {"Q2_K", "DeepSeek-V4-Flash-Vision-Exp-Q2_K-00001-of-00002.gguf", 2, "mmproj-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf", "", "", "", "dspark-DeepSeek-V4-Flash-Vision-Exp-MXFP4.gguf"},
      {"Q2_K_S", "DeepSeek-V4-Flash-Vision-Exp-Q2_K_S-00001-of-00002.gguf", 2, "mmproj-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf", "", "", "", "dspark-DeepSeek-V4-Flash-Vision-Exp-MXFP4.gguf"}},
     {"MXFP4", "Q2_K", "Q2_K_S"}},

    {"ggml-org/gemma-4-E2B-it-GGUF",
     {{"gemma-4-E2B-it-Q8_0.gguf"},
      {"mmproj-gemma-4-E2B-it-BF16.gguf"},
      {"mmproj-gemma-4-E2B-it-Q8_0.gguf"},
      {"mtp-gemma-4-E2B-it-BF16.gguf"},
      {"mtp-gemma-4-E2B-it-Q4_0.gguf"},
      {"mtp-gemma-4-E2B-it-Q8_0.gguf"}},
     {
      {"", "gemma-4-E2B-it-Q8_0.gguf", 1, "mmproj-gemma-4-E2B-it-Q8_0.gguf", "mtp-gemma-4-E2B-it-Q8_0.gguf", "", "", ""},
      {"Q8_0", "gemma-4-E2B-it-Q8_0.gguf", 1, "mmproj-gemma-4-E2B-it-Q8_0.gguf", "mtp-gemma-4-E2B-it-Q8_0.gguf", "", "", ""},
      {"Q4_0", "", 0, "", "mtp-gemma-4-E2B-it-Q4_0.gguf", "", "", ""}},
     {"Q8_0"}},

    {"ggml-org/GLM-4.5V-GGUF",
     {{"GLM-4.5V-Q4_K_M", 2},
      {"mmproj-GLM-4.5V-Q8_0.gguf"},
      {"mtp-GLM-4.5V-Q4_0.gguf"},
      {"mtp-GLM-4.5V-Q8_0.gguf"}},
     {
      {"", "GLM-4.5V-Q4_K_M-00001-of-00002.gguf", 2, "mmproj-GLM-4.5V-Q8_0.gguf", "mtp-GLM-4.5V-Q4_0.gguf", "", "", ""},
      {"Q4_K_M", "GLM-4.5V-Q4_K_M-00001-of-00002.gguf", 2, "mmproj-GLM-4.5V-Q8_0.gguf", "mtp-GLM-4.5V-Q4_0.gguf", "", "", ""}},
     {"Q4_K_M"}},

    {"ggml-org/gpt-oss-120b-GGUF",
     {{"eagle3-gpt-oss-120b-BF16.gguf"},
      {"eagle3-gpt-oss-120b-Q8_0.gguf"},
      {"gpt-oss-120b-MXFP4.gguf"}},
     {
      {"", "gpt-oss-120b-MXFP4.gguf", 1, "", "", "", "eagle3-gpt-oss-120b-Q8_0.gguf", ""},
      {"MXFP4", "gpt-oss-120b-MXFP4.gguf", 1, "", "", "", "eagle3-gpt-oss-120b-Q8_0.gguf", ""}},
     {"MXFP4"}},

    {"ggml-org/Muse-Glimmer-30B-GGUF",
     {{"Muse-Glimmer-30B-Q4_K_M.gguf"},
      {"dflash-Muse-Glimmer-30B-BF16.gguf"},
      {"dflash-Muse-Glimmer-30B-Q4_0.gguf"},
      {"dflash-Muse-Glimmer-30B-Q8_0.gguf"},
      {"mmproj-Muse-Glimmer-30B-BF16.gguf"},
      {"mmproj-Muse-Glimmer-30B-Q8_0.gguf"}},
     {
      {"", "Muse-Glimmer-30B-Q4_K_M.gguf", 1, "mmproj-Muse-Glimmer-30B-Q8_0.gguf", "", "dflash-Muse-Glimmer-30B-Q4_0.gguf", "", ""},
      {"Q4_K_M", "Muse-Glimmer-30B-Q4_K_M.gguf", 1, "mmproj-Muse-Glimmer-30B-Q8_0.gguf", "", "dflash-Muse-Glimmer-30B-Q4_0.gguf", "", ""}},
     {"Q4_K_M"}},

    {"ggml-org/SmolVLM-500M-Instruct-GGUF",
     {{"SmolVLM-500M-Instruct-Q8_0.gguf"},
      {"mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"},
      {"mmproj-SmolVLM-500M-Instruct-f16.gguf"}},
     {
      {"", "SmolVLM-500M-Instruct-Q8_0.gguf", 1, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "SmolVLM-500M-Instruct-Q8_0.gguf", 1, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", "", "", "", ""}},
     {"Q8_0"}},

    {"ggml-org/tinygemma3-GGUF",
     {{"mmproj-tinygemma3.gguf"},
      {"tinygemma3-Q8_0.gguf"}},
     {
      {"", "tinygemma3-Q8_0.gguf", 1, "mmproj-tinygemma3.gguf", "", "", "", ""},
      {"Q8_0", "tinygemma3-Q8_0.gguf", 1, "mmproj-tinygemma3.gguf", "", "", "", ""}},
     {"Q8_0"}},

    {"ggml-org/vocabs",
     {{"HybridDNA/ggml-vocab-carbon.gguf"}},
     {
      {"", "HybridDNA/ggml-vocab-carbon.gguf", 1, "", "", "", "", ""},
      {"CARBON", "HybridDNA/ggml-vocab-carbon.gguf", 1, "", "", "", "", ""}},
     {"CARBON"}},

    {"google/gemma-3-12b-it-qat-q4_0-gguf",
     {{"gemma-3-12b-it-q4_0.gguf"},
      {"mmproj-model-f16-12B.gguf"}},
     {
      {"", "gemma-3-12b-it-q4_0.gguf", 1, "mmproj-model-f16-12B.gguf", "", "", "", ""},
      {"Q4_0", "gemma-3-12b-it-q4_0.gguf", 1, "mmproj-model-f16-12B.gguf", "", "", "", ""}},
     {"Q4_0"}},

    {"google/gemma-4-26B-A4B-it-qat-q4_0-gguf",
     {{"gemma-4-26B-it-mmproj.gguf"},
      {"gemma-4-26B_q4_0-it.gguf"}},
     {
      {"", "gemma-4-26B_q4_0-it.gguf", 1, "gemma-4-26B-it-mmproj.gguf", "", "", "", ""},
      {"IT", "gemma-4-26B_q4_0-it.gguf", 1, "gemma-4-26B-it-mmproj.gguf", "", "", "", ""}},
     {"IT"}},

    {"ibm-granite/granite-vision-3.3-2b-chart2csv-preview-GGUF",
     {{"granite-vision-3.3-2b-chart2csv-preview-Q4_K_M.gguf"},
      {"granite-vision-3.3-2b-chart2csv-preview-Q5_K_M.gguf"},
      {"granite-vision-3.3-2b-chart2csv-preview-Q6_K.gguf"},
      {"granite-vision-3.3-2b-chart2csv-preview-Q8_0.gguf"},
      {"granite-vision-3.3-2b-chart2csv-preview-bf16.gguf"},
      {"mmproj-model-f16.gguf"}},
     {
      {"", "granite-vision-3.3-2b-chart2csv-preview-Q4_K_M.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "granite-vision-3.3-2b-chart2csv-preview-Q4_K_M.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"Q5_K_M", "granite-vision-3.3-2b-chart2csv-preview-Q5_K_M.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"Q6_K", "granite-vision-3.3-2b-chart2csv-preview-Q6_K.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"Q8_0", "granite-vision-3.3-2b-chart2csv-preview-Q8_0.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"BF16", "granite-vision-3.3-2b-chart2csv-preview-bf16.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""}},
     {"BF16", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"}},

    {"ibm-granite/granitelib-rag-r1.0",
     {{"answerability/granite4.1_30b/lora/Lora-bf16.gguf"}},
     {
      {"", "answerability/granite4.1_30b/lora/Lora-bf16.gguf", 1, "", "", "", "", ""},
      {"BF16", "answerability/granite4.1_30b/lora/Lora-bf16.gguf", 1, "", "", "", "", ""}},
     {"BF16"}},

    {"jashepp/Ornith-1.5-35B-A3B-MXFP4_MOE_Hybrid-Imatrix-GGUF",
     {{"Ornith-1.5-35B-A3B-MXFP4_MOE-Only-Imatrix.gguf"},
      {"Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf"},
      {"Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0_F16-Imatrix.gguf"},
      {"z-combined-imatrix.gguf"}},
     {
      {"", "Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf", 1, "", "", "", "", ""},
      {"IMATRIX", "Ornith-1.5-35B-A3B-MXFP4_MOE-Only-Imatrix.gguf", 1, "", "", "", "", ""}},
     {"IMATRIX"}},

    {"JetBrains/Qwen3.8-3.6-27B-blend-GGUF",
     {{"Qwen3.8-3.6-27B-blend-Q4_K_M.gguf"},
      {"mmproj-Qwen3.8-3.6-27B-blend-BF16.gguf"},
      {"mmproj-Qwen3.8-3.6-27B-blend-Q8_0.gguf"},
      {"quantization-mmproj-Q8_0.json"}},
     {
      {"", "Qwen3.8-3.6-27B-blend-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-3.6-27B-blend-Q8_0.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-3.6-27B-blend-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-3.6-27B-blend-Q8_0.gguf", "", "", "", ""},
      {"IQ3_S", "", 0, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"LiquidAI/LFM2-Audio-1.5B-GGUF",
     {{"LFM2-Audio-1.5B-F16.gguf"},
      {"LFM2-Audio-1.5B-Q8_0.gguf"},
      {"audiodecoder-LFM2-Audio-1.5B-F16.gguf"},
      {"audiodecoder-LFM2-Audio-1.5B-Q8_0.gguf"},
      {"mmproj-audioencoder-LFM2-Audio-1.5B-F16.gguf"},
      {"mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf"},
      {"runners/android-arm64/lfm2-audio-android-arm64.zip"},
      {"runners/macos-arm64/lfm2-audio-macos-arm64.zip"},
      {"runners/ubuntu-arm64/lfm2-audio-ubuntu-arm64.zip"},
      {"runners/ubuntu-x64/lfm2-audio-ubuntu-x64.zip"}},
     {
      {"", "LFM2-Audio-1.5B-Q8_0.gguf", 1, "mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf", "", "", "", ""},
      {"F16", "LFM2-Audio-1.5B-F16.gguf", 1, "mmproj-audioencoder-LFM2-Audio-1.5B-F16.gguf", "", "", "", ""},
      {"Q8_0", "LFM2-Audio-1.5B-Q8_0.gguf", 1, "mmproj-audioencoder-LFM2-Audio-1.5B-Q8_0.gguf", "", "", "", ""}},
     {"F16", "Q8_0"}},

    {"LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF",
     {{"LFM2.5-1.2B-Instruct-DSpark-F16.gguf"},
      {"LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf"},
      {"LFM2.5-1.2B-Instruct-DSpark-Q8_0.gguf"}},
     {
      {"", "LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {}},

    {"LiquidAI/LFM2.5-Audio-1.5B-GGUF",
     {{"LFM2.5-Audio-1.5B-F16.gguf"},
      {"LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"LFM2.5-Audio-1.5B-Q8_0.gguf"},
      {"liquid_audio_chat.py"},
      {"mmproj-LFM2.5-Audio-1.5B-F16.gguf"},
      {"mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf"},
      {"runners/llama-liquid-audio-android-arm64.zip"},
      {"runners/llama-liquid-audio-macos-arm64.zip"},
      {"runners/llama-liquid-audio-ubuntu-arm64.zip"},
      {"runners/llama-liquid-audio-ubuntu-x64.zip"},
      {"tokenizer-LFM2.5-Audio-1.5B-F16.gguf"},
      {"tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf"},
      {"vocoder-LFM2.5-Audio-1.5B-F16.gguf"},
      {"vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf"}},
     {
      {"", "LFM2.5-Audio-1.5B-Q8_0.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf", "", "", "", ""},
      {"F16", "LFM2.5-Audio-1.5B-F16.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-F16.gguf", "", "", "", ""},
      {"Q4_0", "LFM2.5-Audio-1.5B-Q4_0.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf", "", "", "", ""},
      {"Q8_0", "LFM2.5-Audio-1.5B-Q8_0.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf", "", "", "", ""}},
     {"F16", "Q4_0", "Q8_0"}},

    {"LiquidAI/LFM2.5-Encoder-230M-GGUF",
     {{"LFM2.5-Encoder-230M-F16.gguf"},
      {"LFM2.5-Encoder-230M-Q4_0.gguf"},
      {"LFM2.5-Encoder-230M-Q8_0.gguf"}},
     {
      {"", "LFM2.5-Encoder-230M-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "LFM2.5-Encoder-230M-F16.gguf", 1, "", "", "", "", ""},
      {"Q4_0", "LFM2.5-Encoder-230M-Q4_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "LFM2.5-Encoder-230M-Q8_0.gguf", 1, "", "", "", "", ""}},
     {"F16", "Q4_0", "Q8_0"}},

    {"MaziyarPanahi/Meta-Llama-3.1-405B-Instruct-GGUF",
     {{"Meta-Llama-3.1-405B-Instruct.Q2_K.gguf", 9}},
     {
      {"", "Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00001-of-00009.gguf", 9, "", "", "", "", ""},
      {"Q2_K", "Meta-Llama-3.1-405B-Instruct.Q2_K.gguf-00001-of-00009.gguf", 9, "", "", "", "", ""}},
     {"Q2_K"}},

    {"MaziyarPanahi/Yi-Coder-9B-Chat-GGUF",
     {{"Yi-Coder-9B-Chat-GGUF_imatrix.dat"},
      {"Yi-Coder-9B-Chat.IQ1_M.gguf"},
      {"Yi-Coder-9B-Chat.Q4_K_M.gguf"}},
     {
      {"", "Yi-Coder-9B-Chat.Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"IQ1_M", "Yi-Coder-9B-Chat.IQ1_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Yi-Coder-9B-Chat.Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {"IQ1_M", "Q4_K_M"}},

    {"microsoft/VibeVoice-ASR-BitNet",
     {{"vibeasr-lm-i2_s-embed-q6_k.gguf"},
      {"vibeasr-vae-encoder-i8_s.gguf"}},
     {
      {"", "vibeasr-lm-i2_s-embed-q6_k.gguf", 1, "", "", "", "", ""},
      {"Q6_K", "vibeasr-lm-i2_s-embed-q6_k.gguf", 1, "", "", "", "", ""},
      {"I8_S", "vibeasr-vae-encoder-i8_s.gguf", 1, "", "", "", "", ""}},
     {"I8_S", "Q6_K"}},

    {"mistralai/Magistral-Small-2506_gguf",
     {{"Magistral-Small-2506.gguf"},
      {"Magistral-Small-2506_Q8_0.gguf"}},
     {
      {"", "Magistral-Small-2506_Q8_0.gguf", 1, "", "", "", "", ""},
      {"2506", "Magistral-Small-2506.gguf", 1, "", "", "", "", ""},
      {"2506_Q8_0", "Magistral-Small-2506_Q8_0.gguf", 1, "", "", "", "", ""}},
     {"2506", "2506_Q8_0"}},

    {"mradermacher/Phi-2-OmniMatrix-i1-GGUF",
     {{"Phi-2-OmniMatrix.i1-IQ1_M.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ1_S.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ2_M.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ2_S.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ2_XS.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ2_XXS.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ3_M.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ3_S.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ3_XS.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ3_XXS.gguf"},
      {"Phi-2-OmniMatrix.i1-IQ4_XS.gguf"},
      {"Phi-2-OmniMatrix.i1-Q2_K.gguf"},
      {"Phi-2-OmniMatrix.i1-Q3_K_L.gguf"},
      {"Phi-2-OmniMatrix.i1-Q3_K_M.gguf"},
      {"Phi-2-OmniMatrix.i1-Q3_K_S.gguf"},
      {"Phi-2-OmniMatrix.i1-Q4_0.gguf"},
      {"Phi-2-OmniMatrix.i1-Q4_0_4_4.gguf"},
      {"Phi-2-OmniMatrix.i1-Q4_0_4_8.gguf"},
      {"Phi-2-OmniMatrix.i1-Q4_0_8_8.gguf"},
      {"Phi-2-OmniMatrix.i1-Q4_K_M.gguf"},
      {"Phi-2-OmniMatrix.i1-Q4_K_S.gguf"},
      {"Phi-2-OmniMatrix.i1-Q5_K_M.gguf"},
      {"Phi-2-OmniMatrix.i1-Q5_K_S.gguf"},
      {"Phi-2-OmniMatrix.i1-Q6_K.gguf"},
      {"imatrix.dat"}},
     {
      {"", "Phi-2-OmniMatrix.i1-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"IQ1_M", "Phi-2-OmniMatrix.i1-IQ1_M.gguf", 1, "", "", "", "", ""},
      {"IQ1_S", "Phi-2-OmniMatrix.i1-IQ1_S.gguf", 1, "", "", "", "", ""},
      {"IQ2_M", "Phi-2-OmniMatrix.i1-IQ2_M.gguf", 1, "", "", "", "", ""},
      {"IQ2_S", "Phi-2-OmniMatrix.i1-IQ2_S.gguf", 1, "", "", "", "", ""},
      {"IQ2_XS", "Phi-2-OmniMatrix.i1-IQ2_XS.gguf", 1, "", "", "", "", ""},
      {"IQ2_XXS", "Phi-2-OmniMatrix.i1-IQ2_XXS.gguf", 1, "", "", "", "", ""},
      {"IQ3_M", "Phi-2-OmniMatrix.i1-IQ3_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Phi-2-OmniMatrix.i1-Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {"IQ1_M", "IQ1_S", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "IQ4_XS", "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K"}},

    {"mradermacher/Qwen3.5-397B-A17B-FP8-GGUF",
     {{"Qwen3.5-397B-A17B-FP8.Q2_K.gguf"},
      {"Qwen3.5-397B-A17B-FP8.mmproj-Q8_0.gguf"},
      {"Qwen3.5-397B-A17B-FP8.mmproj-f16.gguf"}},
     {
      {"", "Qwen3.5-397B-A17B-FP8.Q2_K.gguf", 1, "Qwen3.5-397B-A17B-FP8.mmproj-Q8_0.gguf", "", "", "", ""},
      {"Q2_K", "Qwen3.5-397B-A17B-FP8.Q2_K.gguf", 1, "Qwen3.5-397B-A17B-FP8.mmproj-Q8_0.gguf", "", "", "", ""}},
     {"Q2_K"}},

    {"noctrex/gemma-4-26B-A4B-it-MXFP4_MOE-GGUF",
     {{"gemma-4-26B-A4B-it-MXFP4_MOE.gguf"},
      {"gemma-4-26B-A4B-it-mmproj-BF16.gguf"},
      {"gemma-4-26B-A4B-it-mmproj-F32.gguf"}},
     {
      {"", "gemma-4-26B-A4B-it-MXFP4_MOE.gguf", 1, "gemma-4-26B-A4B-it-mmproj-BF16.gguf", "", "", "", ""},
      {"MXFP4_MOE", "gemma-4-26B-A4B-it-MXFP4_MOE.gguf", 1, "gemma-4-26B-A4B-it-mmproj-BF16.gguf", "", "", "", ""}},
     {"MXFP4_MOE"}},

    {"noctrex/KAT-Coder-V2.5-Dev-MXFP4_MOE-GGUF",
     {{"KAT-Coder-V2.5-Dev-MTP-MXFP4_MOE.gguf"}},
     {
      {"", "KAT-Coder-V2.5-Dev-MTP-MXFP4_MOE.gguf", 1, "", "", "", "", ""},
      {"MXFP4_MOE", "KAT-Coder-V2.5-Dev-MTP-MXFP4_MOE.gguf", 1, "", "", "", "", ""}},
     {"MXFP4_MOE"}},

    {"NousResearch/DeepHermes-AscensionMaze-RLAIF-8b-Atropos-GGUF",
     {{"DeepHermes-AscensionMaze-RLAIF-8b-Atropos-Q4.gguf"},
      {"ascend_cold.gguf"},
      {"ascend_graphmaybe.gguf"},
      {"ascend_lildanger.gguf"},
      {"ascend_retlow.gguf"},
      {"ascend_smartyrude.gguf"},
      {"ascend_smolpost.gguf"}},
     {
      {"", "DeepHermes-AscensionMaze-RLAIF-8b-Atropos-Q4.gguf", 1, "", "", "", "", ""},
      {"Q4", "DeepHermes-AscensionMaze-RLAIF-8b-Atropos-Q4.gguf", 1, "", "", "", "", ""}},
     {"Q4"}},

    {"nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
     {{"nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"}},
     {
      {"", "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf", 1, "", "", "", "", ""},
      {"F16", "nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf", 1, "", "", "", "", ""}},
     {"F16"}},

    // an omni model with a vision and an audio encoder, the plan holds one mmproj
    {"openbmb/MiniCPM-o-4_5-gguf",
     {{"MiniCPM-o-4_5-Q4_K_M.gguf"},
      {"audio/MiniCPM-o-4_5-audio-F16.gguf"},
      {"token2wav-gguf/encoder.gguf"},
      {"token2wav-gguf/flow_extra.gguf"},
      {"token2wav-gguf/flow_matching.gguf"},
      {"token2wav-gguf/hifigan2.gguf"},
      {"token2wav-gguf/prompt_cache.gguf"},
      {"tts/MiniCPM-o-4_5-projector-F16.gguf"},
      {"tts/MiniCPM-o-4_5-tts-F16.gguf"},
      {"vision/MiniCPM-o-4_5-vision-F16.gguf"}},
     {
      {"", "MiniCPM-o-4_5-Q4_K_M.gguf", 1, nullptr, "", "", "", ""},
      {"Q4_K_M", "MiniCPM-o-4_5-Q4_K_M.gguf", 1, nullptr, "", "", "", ""},
      {"F16", "audio/MiniCPM-o-4_5-audio-F16.gguf", 1, nullptr, "", "", "", ""}},
     {"F16", "Q4_K_M"}},

    {"openbmb/MiniCPM-V-4.7-1B-DSpark-gguf",
     {{"dspark-MiniCPM-V-4_7-1B-F16.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q4_0.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q4_1.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q4_K_M.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q4_K_S.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q5_0.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q5_1.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q5_K_M.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q5_K_S.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q6_K.gguf"},
      {"dspark-MiniCPM-V-4_7-1B-Q8_0.gguf"}},
     {
      {"", "dspark-MiniCPM-V-4_7-1B-Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {}},

    {"openbmb/MiniCPM4-0.5B-QAT-Int4-GGUF",
     {{"MiniCPM4-0.5B-QAT-Int4_gptq_aware_q4_0.gguf"}},
     {
      {"", "MiniCPM4-0.5B-QAT-Int4_gptq_aware_q4_0.gguf", 1, "", "", "", "", ""},
      {"INT4_GPTQ_AWARE_Q4_0", "MiniCPM4-0.5B-QAT-Int4_gptq_aware_q4_0.gguf", 1, "", "", "", "", ""}},
     {"INT4_GPTQ_AWARE_Q4_0"}},

    {"openbmb/MiniCPM5-2B-DSpark-GGUF",
     {{"MiniCPM5-2.6B-DSpark.gguf"}},
     {
      {"", "MiniCPM5-2.6B-DSpark.gguf", 1, "", "", "", "", ""}},
     {}},

    {"pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF",
     {{"Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf"}},
     {
      {"", "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf", 1, "", "", "", "", ""},
      {"MTP", "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf", 1, "", "", "", "", ""}},
     {"MTP"}},

    {"Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF",
     {{"Qwen3VL-30B-A3B-Instruct-F16-split", 2},
      {"Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf"},
      {"mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf"},
      {"mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf"}},
     {
      {"", "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf", 1, "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"SPLIT", "Qwen3VL-30B-A3B-Instruct-F16-split-00001-of-00002.gguf", 2, "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf", 1, "mmproj-Qwen3VL-30B-A3B-Instruct-Q8_0.gguf", "", "", "", ""}},
     {"Q4_K_M", "SPLIT"}},

    {"Qwen/QwQ-32B-GGUF",
     {{"fp16/qwq-32b-fp16", 17},
      {"qwq-32b-q4_k_m.gguf"}},
     {
      {"", "qwq-32b-q4_k_m.gguf", 1, "", "", "", "", ""},
      {"FP16", "fp16/qwq-32b-fp16-00001-of-00017.gguf", 17, "", "", "", "", ""},
      {"Q4_K_M", "qwq-32b-q4_k_m.gguf", 1, "", "", "", "", ""}},
     {"FP16", "Q4_K_M"}},

    {"second-state/gpt-oss-120b-GGUF",
     {{"gpt-oss-120b-MXFP4_MOE", 3},
      {"gpt-oss-120b-f16", 3}},
     {
      {"", "gpt-oss-120b-MXFP4_MOE-00001-of-00003.gguf", 3, "", "", "", "", ""},
      {"MXFP4_MOE", "gpt-oss-120b-MXFP4_MOE-00001-of-00003.gguf", 3, "", "", "", "", ""},
      {"F16", "gpt-oss-120b-f16-00001-of-00003.gguf", 3, "", "", "", "", ""}},
     {"F16", "MXFP4_MOE"}},

    {"second-state/Llava-v1.5-7B-GGUF",
     {{"llava-v1.5-7b-Q4_K_M.gguf"},
      {"llava-v1.5-7b-mmproj-model-f16.gguf"}},
     {
      {"", "llava-v1.5-7b-Q4_K_M.gguf", 1, "llava-v1.5-7b-mmproj-model-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "llava-v1.5-7b-Q4_K_M.gguf", 1, "llava-v1.5-7b-mmproj-model-f16.gguf", "", "", "", ""}},
     {"Q4_K_M"}},

    {"second-state/Qwen2-Audio-7B-Instruct-GGUF",
     {{"Qwen2-Audio-7B-Instruct-Q2_K.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q3_K_L.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q3_K_M.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q3_K_S.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q4_0.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q4_K_M.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q4_K_S.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q5_0.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q5_K_M.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q5_K_S.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q6_K.gguf"},
      {"Qwen2-Audio-7B-Instruct-Q8_0.gguf"},
      {"Qwen2-Audio-7B-Instruct-f16.gguf"}},
     {
      {"", "Qwen2-Audio-7B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "Qwen2-Audio-7B-Instruct-Q2_K.gguf", 1, "", "", "", "", ""},
      {"Q3_K_L", "Qwen2-Audio-7B-Instruct-Q3_K_L.gguf", 1, "", "", "", "", ""},
      {"Q3_K_M", "Qwen2-Audio-7B-Instruct-Q3_K_M.gguf", 1, "", "", "", "", ""},
      {"Q3_K_S", "Qwen2-Audio-7B-Instruct-Q3_K_S.gguf", 1, "", "", "", "", ""},
      {"Q4_0", "Qwen2-Audio-7B-Instruct-Q4_0.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen2-Audio-7B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_S", "Qwen2-Audio-7B-Instruct-Q4_K_S.gguf", 1, "", "", "", "", ""}},
     {"F16", "Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"}},

    {"second-state/Qwen2-VL-2B-Instruct-GGUF",
     {{"Qwen2-VL-2B-Instruct-Q4_K_M.gguf"},
      {"Qwen2-VL-2B-Instruct-vision-encoder.gguf"}},
     {
      {"", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", 1, "Qwen2-VL-2B-Instruct-vision-encoder.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen2-VL-2B-Instruct-Q4_K_M.gguf", 1, "Qwen2-VL-2B-Instruct-vision-encoder.gguf", "", "", "", ""}},
     {"Q4_K_M"}},

    {"second-state/Qwen2.5-VL-7B-Instruct-GGUF",
     {{"Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"},
      {"Qwen2.5-VL-7B-Instruct-vision.gguf"}},
     {
      {"", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 1, "Qwen2.5-VL-7B-Instruct-vision.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 1, "Qwen2.5-VL-7B-Instruct-vision.gguf", "", "", "", ""}},
     {"Q4_K_M"}},

    {"stepfun-ai/Step-3.5-Flash-GGUF-Q4_K_S",
     {{"step3p5_flash_Q4_K_S", 12}},
     {
      {"", "step3p5_flash_Q4_K_S-00001-of-00012.gguf", 12, "", "", "", "", ""}},
     {"Q4_K_S"}},

    {"stepfun-ai/Step-3.7-Flash-GGUF",
     {{"BF16/Step3.7-flash-bf16", 9},
      {"Q8_0/Step-3.7-flash-Q8_0", 5},
      {"Step-3.7.imatrix.gguf"},
      {"Step3.7-flash-mtp-BF16.gguf"},
      {"Step3.7-flash-mtp-Q8_0.gguf"},
      {"mmproj-step3.7-flash-f16.gguf"}},
     {
      {"", "Q8_0/Step-3.7-flash-Q8_0-00001-of-00005.gguf", 5, "mmproj-step3.7-flash-f16.gguf", "Step3.7-flash-mtp-Q8_0.gguf", "", "", ""},
      {"BF16", "BF16/Step3.7-flash-bf16-00001-of-00009.gguf", 9, "mmproj-step3.7-flash-f16.gguf", "Step3.7-flash-mtp-BF16.gguf", "", "", ""},
      {"Q8_0", "Q8_0/Step-3.7-flash-Q8_0-00001-of-00005.gguf", 5, "mmproj-step3.7-flash-f16.gguf", "Step3.7-flash-mtp-Q8_0.gguf", "", "", ""}},
     {"BF16", "Q8_0"}},

    {"TheBloke/NeuralBeagle14-7B-GGUF",
     {{"neuralbeagle14-7b.Q2_K.gguf"},
      {"neuralbeagle14-7b.Q3_K_L.gguf"},
      {"neuralbeagle14-7b.Q3_K_M.gguf"},
      {"neuralbeagle14-7b.Q3_K_S.gguf"},
      {"neuralbeagle14-7b.Q4_0.gguf"},
      {"neuralbeagle14-7b.Q4_K_M.gguf"},
      {"neuralbeagle14-7b.Q4_K_S.gguf"},
      {"neuralbeagle14-7b.Q5_0.gguf"},
      {"neuralbeagle14-7b.Q5_K_M.gguf"},
      {"neuralbeagle14-7b.Q5_K_S.gguf"},
      {"neuralbeagle14-7b.Q6_K.gguf"},
      {"neuralbeagle14-7b.Q8_0.gguf"}},
     {
      {"", "neuralbeagle14-7b.Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "neuralbeagle14-7b.Q2_K.gguf", 1, "", "", "", "", ""},
      {"Q3_K_L", "neuralbeagle14-7b.Q3_K_L.gguf", 1, "", "", "", "", ""},
      {"Q3_K_M", "neuralbeagle14-7b.Q3_K_M.gguf", 1, "", "", "", "", ""},
      {"Q3_K_S", "neuralbeagle14-7b.Q3_K_S.gguf", 1, "", "", "", "", ""},
      {"Q4_0", "neuralbeagle14-7b.Q4_0.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "neuralbeagle14-7b.Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_S", "neuralbeagle14-7b.Q4_K_S.gguf", 1, "", "", "", "", ""}},
     {"Q2_K", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"}},

    {"ubergarm/Kimi-K2-Instruct-GGUF",
     {{"imatrix-Kimi-K2-Instruct-Q8_0.dat"},
      {"mainline/imatrix-mainline-pr9400-plus-kimi-k2-942c55cd5-Kimi-K2-Instruct-Q8_0.gguf"},
      {"smol-IQ1_KT/Kimi-K2-Instruct-smol-IQ1_KT", 5}},
     {
      {"", "smol-IQ1_KT/Kimi-K2-Instruct-smol-IQ1_KT-00001-of-00005.gguf", 5, "", "", "", "", ""},
      {"IQ1_KT", "smol-IQ1_KT/Kimi-K2-Instruct-smol-IQ1_KT-00001-of-00005.gguf", 5, "", "", "", "", ""}},
     {"IQ1_KT"}},

    {"unsloth/DeepSeek-V4-Flash-0731-GGUF",
     {{"UD-IQ1_M/DeepSeek-V4-Flash-0731-UD-IQ1_M", 3},
      {"dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
      {"dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf"}},
     {
      {"", "UD-IQ1_M/DeepSeek-V4-Flash-0731-UD-IQ1_M-00001-of-00003.gguf", 3, "", "", "", "", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
      {"IQ1_M", "UD-IQ1_M/DeepSeek-V4-Flash-0731-UD-IQ1_M-00001-of-00003.gguf", 3, "", "", "", "", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"}},
     {"IQ1_M"}},

    {"unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF",
     {{"UD-IQ1_M/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_M", 3},
      {"UD-IQ1_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_S", 3},
      {"UD-IQ2_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ2_XXS", 3},
      {"UD-IQ3_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_S", 4},
      {"UD-IQ3_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS", 4},
      {"UD-IQ4_XS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ4_XS", 4},
      {"UD-Q2_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL", 3},
      {"UD-Q3_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q3_K_XL", 4},
      {"UD-Q4_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q4_K_XL", 5},
      {"UD-Q8_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q8_K_XL", 5},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"}},
     {
      {"", "UD-IQ1_M/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_M-00001-of-00003.gguf", 3, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ1_M", "UD-IQ1_M/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_M-00001-of-00003.gguf", 3, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ1_S", "UD-IQ1_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ1_S-00001-of-00003.gguf", 3, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ2_XXS", "UD-IQ2_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ2_XXS-00001-of-00003.gguf", 3, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ3_S", "UD-IQ3_S/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_S-00001-of-00004.gguf", 4, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ3_XXS", "UD-IQ3_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf", 4, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ4_XS", "UD-IQ4_XS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ4_XS-00001-of-00004.gguf", 4, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q2_K_XL", "UD-Q2_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf", 3, "mmproj-BF16.gguf", "", "", "", ""}},
     {"IQ1_M", "IQ1_S", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_XS", "Q2_K_XL", "Q3_K_XL", "Q4_K_XL", "Q8_K_XL"}},

    {"unsloth/gemma-4-E2B-it-GGUF",
     {{"MTP/mtp-gemma-4-E2B-it-BF16.gguf"},
      {"MTP/mtp-gemma-4-E2B-it-F16.gguf"},
      {"MTP/mtp-gemma-4-E2B-it-Q8_0.gguf"},
      {"gemma-4-E2B-it-Q4_K_M.gguf"},
      {"imatrix_unsloth.gguf_file"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"},
      {"mtp-gemma-4-E2B-it.gguf"}},
     {
      {"", "gemma-4-E2B-it-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "mtp-gemma-4-E2B-it.gguf", "", "", ""},
      {"Q4_K_M", "gemma-4-E2B-it-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "mtp-gemma-4-E2B-it.gguf", "", "", ""}},
     {"Q4_K_M"}},

    {"unsloth/LFM2.5-24B-A2B-GGUF",
     {{"BF16/LFM2.5-24B-A2B-BF16.gguf"},
      {"LFM2.5-24B-A2B-Q8_0.gguf"}},
     {
      {"", "LFM2.5-24B-A2B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "BF16/LFM2.5-24B-A2B-BF16.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "LFM2.5-24B-A2B-Q8_0.gguf", 1, "", "", "", "", ""}},
     {"BF16", "Q8_0"}},

    {"unsloth/Muse-Glimmer-30B-GGUF",
     {{"Muse-Glimmer-30B-Q8_0.gguf"},
      {"dflash-kquant.gguf"},
      {"mmproj-Muse-Glimmer-30B-BF16.gguf"},
      {"mmproj-Muse-Glimmer-30B-Q8_0.gguf"},
      {"mmproj-kquant.gguf"}},
     {
      {"", "Muse-Glimmer-30B-Q8_0.gguf", 1, "mmproj-Muse-Glimmer-30B-Q8_0.gguf", "", "dflash-kquant.gguf", "", ""},
      {"Q8_0", "Muse-Glimmer-30B-Q8_0.gguf", 1, "mmproj-Muse-Glimmer-30B-Q8_0.gguf", "", "dflash-kquant.gguf", "", ""}},
     {"Q8_0"}},

    {"unsloth/Qwen3.8-27B-GGUF",
     {{"MTP/mtp-Qwen3.8-27B-Q4_0.gguf"},
      {"Qwen3.8-27B-UD-IQ1_M.gguf"},
      {"Qwen3.8-27B-UD-Q4_K_M.gguf"},
      {"imatrix_unsloth.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"}},
     {
      {"", "Qwen3.8-27B-UD-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "", "", ""},
      {"IQ1_M", "Qwen3.8-27B-UD-IQ1_M.gguf", 1, "mmproj-BF16.gguf", "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-UD-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "", "", ""}},
     {"IQ1_M", "Q4_K_M"}},
};

static std::vector<std::string> real_files(const real_repo & r) {
    std::vector<std::string> out;
    for (const auto & f : r.files) {
        if (f.parts == 0) {
            out.push_back(f.name);
            continue;
        }
        for (int i = 1; i <= f.parts; ++i) {
            char suffix[32];
            snprintf(suffix, sizeof(suffix), "-%05d-of-%05d.gguf", i, f.parts);
            out.push_back(std::string(f.name) + suffix);
        }
    }
    return out;
}

// every mismatch is reported before the test fails, so one run lists them all
static int g_real_failures = 0;

static void expect_eq(const std::string & what, const std::string & actual, const char * expected) {
    if (expected && actual != expected) {
        fprintf(stderr, "  [%s] %s\n    actual:   '%s'\n    expected: '%s'\n",
                g_context.c_str(), what.c_str(), actual.c_str(), expected);
        g_real_failures++;
    }
}

static void check_real_plan(const real_repo & r, const real_plan & p) {
    g_context = std::string(r.id) + (p.tag[0] ? std::string(":") + p.tag : "");

    common_download_opts opts;
    opts.download_mmproj = true;
    opts.download_mtp    = true;
    opts.download_dflash = true;
    opts.download_eagle3 = true;
    opts.download_dspark = true;

    auto plan = common_download_get_hf_plan(model_ref(g_context), opts);

    expect_eq("primary", plan.primary.path, p.primary);
    expect_eq("parts",   std::to_string(plan.model_files.size()), std::to_string(p.parts).c_str());
    expect_eq("mmproj",  plan.mmproj.path, p.mmproj);
    expect_eq("mtp",     plan.mtp.path,    p.mtp);
    expect_eq("dflash",  plan.dflash.path, p.dflash);
    expect_eq("eagle3",  plan.eagle3.path, p.eagle3);
    expect_eq("dspark",  plan.dspark.path, p.dspark);
}

// the listing reads the cache, so the repo is materialized there as a snapshot
static void check_real_listing(const real_repo & r, const std::vector<std::string> & files) {
    g_context = std::string(r.id) + " listing";

    std::string folder = r.id;
    string_replace_all(folder, "/", "--");
    const std::filesystem::path repo_dir = cache_dir / ("models--" + folder);
    std::filesystem::create_directories(repo_dir / "refs");
    { std::ofstream(repo_dir / "refs" / "main") << COMMIT; }
    for (const auto & f : files) {
        std::filesystem::path p = cached(r.id, f);
        std::filesystem::create_directories(p.parent_path());
        std::ofstream(p) << "gguf";
    }

    std::vector<std::string> actual;
    for (const auto & e : common_list_cached_models()) {
        if (e.repo == r.id) {
            actual.push_back(e.tag);
        }
    }
    std::sort(actual.begin(), actual.end());
    auto expected = r.listing;
    std::sort(expected.begin(), expected.end());
    expect_eq("tags", string_join(actual, " "), string_join(expected, " ").c_str());

    std::filesystem::remove_all(repo_dir);
}

static void test_real_repos() {
    printf("test-model-resolution: real listings on %zu repos\n", sizeof(real_repos) / sizeof(real_repos[0]));

    for (const auto & r : real_repos) {
        auto files = real_files(r);
        g_repos.clear();
        g_repos[r.id] = files;
        for (const auto & p : r.plans) {
            check_real_plan(r, p);
        }
        check_real_listing(r, files);
    }
    g_repos.clear();

    if (g_real_failures > 0) {
        fprintf(stderr, "test-model-resolution: %d mismatches on real listings\n", g_real_failures);
        std::abort();
    }
}

//
// end-to-end assembly: real CLI parsing, real handler init resolving over the
// loopback, downloads skipped by flipping offline before apply
//

static void assemble(std::vector<std::string> argv, common_params & params, llama_example ex = LLAMA_EXAMPLE_SERVER) {
    std::vector<char *> cargv;
    g_context.clear();
    for (auto & a : argv) {
        g_context += g_context.empty() ? a : " " + a;
        cargv.push_back(a.data());
    }
    bool ok = common_params_parse((int) cargv.size(), cargv.data(), params, ex);
    REQUIRE(ok);

    auto handler = common_models_handler_init(params, ex);

    // skip the network execution, on_done still wires the params
    params.offline = true;
    common_models_handler_apply(handler, params);
}

static void test_task_assembly() {
    printf("test-model-resolution: end-to-end assembly\n");

    g_repos["test/main"]   = flat;
    g_repos["test/hole"]   = hole;
    g_repos["test/quad"]   = quad;
    g_repos["test/dflash"] = dflash_only;
    g_repos["test/eagle3"] = eagle3_only;
    g_repos["test/spark"]  = spark;
    g_repos["test/pair"]   = dspark_dflash;
    g_repos["test/small"]  = {"draft-model-Q4_K_M.gguf"};
    g_repos["test/preset"] = {"preset.ini", "model-Q8_0.gguf"};

    {
        // plain -hf wires the model and its mmproj, nothing speculative
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0"}, params);
        REQUIRE_EQ(params.model.path,  cached("test/main", "model-Q8_0.gguf"));
        REQUIRE_EQ(params.mmproj.path, cached("test/main", "mmproj-model-Q8_0.gguf"));
        REQUIRE(params.speculative.draft.mparams.path.empty());
    }
    {
        // plain -hf wires the model and its mmproj, nothing speculative
        common_params params;
        assemble({"download", "-hf", "test/main:Q8_0"}, params, LLAMA_EXAMPLE_DOWNLOAD);
        REQUIRE_EQ(params.model.path,  cached("test/main", "model-Q8_0.gguf"));
        REQUIRE_EQ(params.mmproj.path, cached("test/main", "mmproj-model-Q8_0.gguf"));
        REQUIRE(params.speculative.draft.mparams.path.empty());
    }
    {
        // --no-mmproj disables the mmproj discovery
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--no-mmproj"}, params);
        REQUIRE(params.mmproj.path.empty());
    }
    {
        // --no-mmproj disables the mmproj discovery
        common_params params;
        assemble({"download", "-hf", "test/main:Q8_0", "--no-mmproj"}, params, LLAMA_EXAMPLE_DOWNLOAD);
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
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/main", "mtp-model-Q8_0.gguf"));
    }
    {
        // -hfd with a spec type wires the draft repo sidecar at its tag,
        // not its full model, and suppresses the main repo fallback
        common_params params;
        assemble({"server", "-hf", "test/hole:Q8_0", "-hfd", "test/hole:Q4_0", "--spec-type", "draft-mtp"}, params);
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/hole", "mtp-model-Q4_0.gguf"));
    }
    {
        // an explicit -md file wins over the sidecar resolution
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/main", "-md", "mtp-model-BF16.gguf", "--spec-type", "draft-mtp"}, params);
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/main", "mtp-model-BF16.gguf"));
    }
    {
        // -hfd without a spec type auto-selects the type, mtp first when all ship
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/quad:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP});
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/quad", "mtp-model-Q8_0.gguf"));
    }
    {
        // auto-selection with only a dflash sidecar
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/dflash:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH});
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/dflash", "dflash-model-Q8_0.gguf"));
    }
    {
        // auto-selection with only an eagle3 sidecar
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/eagle3:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3});
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/eagle3", "eagle3-model-Q8_0.gguf"));
    }
    {
        // auto-selection prefers dspark over dflash when both ship
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/pair:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK});
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/pair", "dspark-model-Q8_0.gguf"));
    }
    {
        // -hf with the dspark spec type wires the sidecar of the main repo,
        // anchored on the only full quant
        common_params params;
        assemble({"server", "-hf", "test/spark", "--spec-type", "draft-dspark"}, params);
        REQUIRE_EQ(params.model.path, cached("test/spark", "model-MXFP4.gguf"));
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/spark", "dspark-model-MXFP4.gguf"));
    }
    {
        // -hfd on a repo without sidecars keeps resolving a full model as draft
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/small"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE});
        REQUIRE_EQ(params.speculative.draft.mparams.path, cached("test/small", "draft-model-Q4_K_M.gguf"));
    }
    {
        // a preset repo wires the preset and clears the model for router mode
        common_params params;
        assemble({"server", "-hf", "test/preset"}, params);
        REQUIRE_EQ(params.models_preset, cached("test/preset", "preset.ini"));
        REQUIRE(params.model.path.empty());
        REQUIRE(params.model.hf_repo.empty());
    }

    g_repos.clear();
}

int main(void) {
    // unbuffered, so a crash cannot swallow the reports already printed
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // the negative cases legitimately log errors on every reordering,
    // keep the output down to the reports
    common_log_pause(common_log_main());

    // the loopback endpoint also keeps the client init from rejecting
    // https on the builds without TLS support
    httplib::Server server;
    serve_repos(server);
    int port = server.bind_to_any_port("127.0.0.1");

    // isolate the cache, its location is read once so it is set
    // before anything else
    cache_dir = std::filesystem::temp_directory_path() /
                ("test-model-resolution-cache-" + std::to_string(port));
    std::filesystem::remove_all(cache_dir);
    common_set_env("LLAMA_CACHE", cache_dir.string());

    std::thread server_thread([&server] { server.listen_after_bind(); });
    server.wait_until_ready();
    common_set_env("MODEL_ENDPOINT", "http://127.0.0.1:" + std::to_string(port) + "/");

    test_plan_resolution();
    test_plan_partials();
    test_sidecar_removal();
    test_shared_removal();
    test_real_repos();
    test_task_assembly();

    server.stop();
    server_thread.join();

    std::filesystem::remove_all(cache_dir);
    printf("test-model-resolution: all tests OK\n");
    return 0;
}
