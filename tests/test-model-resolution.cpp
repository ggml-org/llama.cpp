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

static const struct {
    const char * repo;
    std::string token;
    bool valid_header;
} token_cases[] = {
    {"test/token-pat",    "hf_" + std::string(34, 'a'),       true},
    {"test/token-oauth",  "hf_oauth_" + std::string(34, 'b'), true},
    {"test/token-short",  "opaque-token",                   true},
    {"test/token-long",   "hf_" + std::string(300, 'c'),      true},
    {"test/token-empty",  "",                               true},
    {"test/token-crlf",   "token\r\nX-Injected: yes",        false},
    {"test/token-null",   std::string("token\0suffix", 12),  false},
};

static bool check_token_header(const httplib::Request & req, httplib::Response & res) {
    for (const auto & c : token_cases) {
        if (req.matches[1].str() != c.repo) {
            continue;
        }
        REQUIRE(c.valid_header);
        const std::string expected = c.token.empty() ? "" : "Bearer " + c.token;
        if (req.get_header_value_count("Authorization") != (c.token.empty() ? 0u : 1u) ||
            req.get_header_value("Authorization") != expected) {
            res.status = 401;
            return false;
        }
    }
    return true;
}

// the server lives in main, so its destructor runs before the static teardown
// tears down the winsock state httplib brings in
static void serve_repos(httplib::Server & server) {
    server.Get(R"(/api/models/(.+)/refs)", [](const httplib::Request & req, httplib::Response & res) {
        if (!check_token_header(req, res)) {
            return;
        }
        if (g_repos.count(req.matches[1])) {
            res.set_content(common_json{{"branches", common_json::array({ common_json{{"name", "main"}, {"targetCommit", COMMIT}} })}}.dump(),
                            "application/json");
        } else {
            res.status = 404;
        }
    });
    server.Get(R"(/api/models/(.+)/tree/.+)", [](const httplib::Request & req, httplib::Response & res) {
        if (!check_token_header(req, res)) {
            return;
        }
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

static void test_token_forwarding() {
    printf("test-model-resolution: token forwarding\n");

    for (const auto & c : token_cases) {
        g_context = c.repo;
        common_download_opts opts;
        opts.bearer_token = c.token;
        const auto plan = common_download_get_hf_plan(model_ref(c.repo), opts);
        if (c.valid_header) {
            REQUIRE_EQ(plan.primary.path, "model-Q4_0.gguf");
            REQUIRE(plan.model_files.size() == 1);
        } else {
            REQUIRE(plan.primary.path.empty());
            REQUIRE(plan.model_files.empty());
        }
    }
    g_repos.clear();
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
// real repo listings captured from the HF API: GGUF repos whose file names follow
// the GGUF naming convention, one listing per layout pattern, each pinned to the
// plans and the cache listing the resolver gives today, so any change in
// resolution shows up here
//

// a file name, or with `parts` a shard prefix standing for the verbatim names
// <prefix>-00001-of-0000n.gguf .. <prefix>-0000n-of-0000n.gguf
struct real_file {
    const char * name;
    int          parts = 0;
};

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
    {"0bserverx/Muse-Glimmer-30B-Heretic-Uncensored-GGUF",
     {{"Muse-Glimmer-30B-Heretic-BF16.gguf"},
      {"Muse-Glimmer-30B-Heretic-IQ4_XS.gguf"},
      {"Muse-Glimmer-30B-Heretic-Q4_K_M.gguf"},
      {"Muse-Glimmer-30B-Heretic-Q8_0.gguf"},
      {"Muse-Glimmer-30B-Heretic-TQ2_0.gguf"}},
     {
      {"", "Muse-Glimmer-30B-Heretic-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Muse-Glimmer-30B-Heretic-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Muse-Glimmer-30B-Heretic-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "Muse-Glimmer-30B-Heretic-BF16.gguf", 1, "", "", "", "", ""},
      {"TQ2_0", "Muse-Glimmer-30B-Heretic-TQ2_0.gguf", 1, "", "", "", "", ""}},
     {"TQ2_0", "Q8_0", "Q4_K_M", "IQ4_XS", "BF16"}},

    {"0bserverx/RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored-GGUF",
     {{"RVN-Qwen3.8-Flash-Next-IQ3_M", 8},
      {"RVN-Qwen3.8-Flash-Next-Q4_0", 8},
      {"RVN-Qwen3.8-Flash-Next-Q4_K_M", 8},
      {"RVN-Qwen3.8-Flash-Next-Q6_K", 8}},
     {
      {"", "RVN-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00008.gguf", 8, "", "", "", "", ""},
      {"Q4_K_M", "RVN-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00008.gguf", 8, "", "", "", "", ""},
      {"IQ3_M", "RVN-Qwen3.8-Flash-Next-IQ3_M-00001-of-00008.gguf", 8, "", "", "", "", ""},
      {"Q6_K", "RVN-Qwen3.8-Flash-Next-Q6_K-00001-of-00008.gguf", 8, "", "", "", "", ""}},
     {"Q6_K", "Q4_K_M", "Q4_0", "IQ3_M"}},

    {"0xSojalSec/Qwen3.8-Flash-Next-Uncensored-GSQ-RCO-12-GB-GPU",
     {{"IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS", 2},
      {"IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_XXS", 2},
      {"Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0", 2},
      {"mmproj-Qwen3.8-Flash-Next-BF16.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q8_0.gguf"},
      {"mtp-q2_0.gguf"}},
     {
      {"", "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mtp-q2_0.gguf", "", "", ""},
      {"Q8_0", "", 0, "", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"IQ2_XS", "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mtp-q2_0.gguf", "", "", ""},
      {"Q2_0", "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mtp-q2_0.gguf", "", "", ""},
      {"BF16", "", 0, "", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""}},
     {"Q2_0", "IQ3_XXS", "IQ2_XS"}},

    {"agentionai/Qwen3.8-Flash-Next-AP-GGUF",
     {{"AP-IQ2_S/Qwen3.8-Flash-Next-AP-IQ2_S.gguf"},
      {"AP-Q4_K_M/Qwen3.8-Flash-Next-AP-Q4_K_M.gguf"},
      {"AP-Q5_K_XL/Qwen3.8-Flash-Next-AP-Q5_K_XL.gguf"},
      {"mmproj-F16.gguf"}},
     {
      {"", "AP-Q4_K_M/Qwen3.8-Flash-Next-AP-Q4_K_M.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"Q4_K_M", "AP-Q4_K_M/Qwen3.8-Flash-Next-AP-Q4_K_M.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"IQ2_S", "AP-IQ2_S/Qwen3.8-Flash-Next-AP-IQ2_S.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"Q5_K_XL", "AP-Q5_K_XL/Qwen3.8-Flash-Next-AP-Q5_K_XL.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q5_K_XL", "Q4_K_M", "IQ2_S"}},

    {"agentionai/Signal-3.8-27B-GGUF",
     {{"AP-IQ3_S/Signal-3.8-27B-AP-IQ3_S.gguf"},
      {"AP-Q4_K_M/Signal-3.8-27B-AP-Q4_K_M.gguf"},
      {"Q8_0/Signal-3.8-27B-Q8_0.gguf"},
      {"mmproj-BF16.gguf"}},
     {
      {"", "AP-Q4_K_M/Signal-3.8-27B-AP-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "AP-Q4_K_M/Signal-3.8-27B-AP-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/Signal-3.8-27B-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ3_S", "AP-IQ3_S/Signal-3.8-27B-AP-IQ3_S.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "IQ3_S"}},

    {"agentionai/Signal-3.8-Flash-Next-GGUF",
     {{"AP-IQ4_XS/Signal-3.8-Flash-Next-AP-IQ4_XS.gguf"},
      {"AP-Q4_K_XL/Signal-3.8-Flash-Next-AP-Q4_K_XL.gguf"},
      {"Q8_0/Signal-3.8-Flash-Next-Q8_0.gguf"},
      {"mmproj-F16.gguf"}},
     {
      {"", "Q8_0/Signal-3.8-Flash-Next-Q8_0.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/Signal-3.8-Flash-Next-Q8_0.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"IQ4_XS", "AP-IQ4_XS/Signal-3.8-Flash-Next-AP-IQ4_XS.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_XL", "IQ4_XS"}},

    {"ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-GGUF",
     {{"RCO/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-RCO-IQ2_S.gguf"},
      {"Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q3_K_XL.gguf"},
      {"Swift-1.5-Qwen3.8-27B-Uncensored-MTP-BF16.gguf"},
      {"mmproj-BF16.gguf"}},
     {
      {"", "RCO/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-RCO-IQ2_S.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ2_S", "RCO/Swift-1.5-Qwen3.8-27B-Uncensored-Dynamic-MTP-RCO-IQ2_S.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "Swift-1.5-Qwen3.8-27B-Uncensored-MTP-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"BF16", "Q3_K_XL", "IQ2_S"}},

    {"ajgazin/Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-GGUF",
     {{"Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-IQ3_S.gguf"},
      {"Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-Q4_K_XL.gguf"},
      {"Swift-Qwen3.8-27B-Uncensored-MTP-BF16.gguf"},
      {"mmproj-BF16.gguf"}},
     {
      {"", "Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-IQ3_S.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-IQ3_S", "Swift-Qwen3.8-27B-Uncensored-Dynamic-MTP-UD-IQ3_S.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "Swift-Qwen3.8-27B-Uncensored-MTP-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"BF16", "Q4_K_XL", "IQ3_S"}},

    {"AliceThirty/GLM-5.3-Flash-UNCENSORED-GGUF",
     {{"GLM-5.3-Flash-UNCENSORED-FP8-mmproj-bf16.gguf"},
      {"UD-Q3_K_XL/GLM-5.3-Flash-UNCENSORED-FP8-UD-Q3_K_XL", 4},
      {"UD-Q5_K_XL/GLM-5.3-Flash-UNCENSORED-FP8-UD-Q5_K_XL", 5}},
     {
      {"", "UD-Q3_K_XL/GLM-5.3-Flash-UNCENSORED-FP8-UD-Q3_K_XL-00001-of-00004.gguf", 4, "GLM-5.3-Flash-UNCENSORED-FP8-mmproj-bf16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""},
      {"UD-Q5_K_XL", "UD-Q5_K_XL/GLM-5.3-Flash-UNCENSORED-FP8-UD-Q5_K_XL-00001-of-00005.gguf", 5, "GLM-5.3-Flash-UNCENSORED-FP8-mmproj-bf16.gguf", "", "", "", ""}},
     {"Q5_K_XL", "Q3_K_XL"}},

    {"antirez/glm-5.3-gguf",
     {{"GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"}},
     {
      {"", "GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf", 1, "", "", "", "", ""},
      {"UD-IQ2_XXS_ROUTEDIQ2XXS_BLK78Q2K", "GLM-5.3-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf", 1, "", "", "", "", ""}},
     {"IQ2_XXS_ROUTEDIQ2XXS_BLK78Q2K"}},

    {"AtomicChat/Qwen-Image-2.1-Turbo-GGUF",
     {{"Qwen-Image-2.1-Turbo-AD-Q2_K.gguf"},
      {"Qwen-Image-2.1-Turbo-AD-Q5_K.gguf"},
      {"Qwen-Image-2.1-Turbo-Q8_0.gguf"}},
     {
      {"", "Qwen-Image-2.1-Turbo-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Qwen-Image-2.1-Turbo-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "Qwen-Image-2.1-Turbo-AD-Q2_K.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_K", "Q2_K"}},

    {"AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF",
     {{"Qwen-Image-2.1-Turbo-Uncensored-AD-Q2_K.gguf"},
      {"Qwen-Image-2.1-Turbo-Uncensored-AD-Q5_K.gguf"},
      {"Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf"},
      {"mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf"},
      {"mmproj-Qwen-Image-2.1-Turbo-Uncensored-F16.gguf"}},
     {
      {"", "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf", 1, "mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf", 1, "mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf", "", "", "", ""},
      {"Q2_K", "Qwen-Image-2.1-Turbo-Uncensored-AD-Q2_K.gguf", 1, "mmproj-Qwen-Image-2.1-Turbo-Uncensored-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q5_K", "Q2_K"}},

    {"audio-cpp/Audio-Flamingo-GGUF",
     {{"audio-flamingo-3-bf16.gguf"},
      {"audio-flamingo-3-q4_k.gguf"},
      {"audio-flamingo-3-q8_0.gguf"},
      {"audio-flamingo-next-bf16.gguf"},
      {"audio-flamingo-next-q4_k.gguf"},
      {"audio-flamingo-next-q8_0.gguf"}},
     {
      {"", "audio-flamingo-3-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "audio-flamingo-3-q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "audio-flamingo-3-bf16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K", "BF16"}},

    {"bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
     {{"DeepSeek-Coder-V2-Lite-Instruct-IQ2_M.gguf"},
      {"DeepSeek-Coder-V2-Lite-Instruct-Q4_K_L.gguf"},
      {"DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"},
      {"DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf"},
      {"DeepSeek-Coder-V2-Lite-Instruct-f32.gguf/DeepSeek-Coder-V2-Lite-Instruct-f32", 2}},
     {
      {"", "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "DeepSeek-Coder-V2-Lite-Instruct-Q8_0.gguf", 1, "", "", "", "", ""},
      {"IQ2_M", "DeepSeek-Coder-V2-Lite-Instruct-IQ2_M.gguf", 1, "", "", "", "", ""},
      {"F32", "DeepSeek-Coder-V2-Lite-Instruct-f32.gguf/DeepSeek-Coder-V2-Lite-Instruct-f32-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"F32", "Q8_0", "Q4_K_M", "Q4_K_L", "IQ2_M"}},

    {"bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
     {{"DeepSeek-R1-Distill-Qwen-14B-IQ2_M.gguf"},
      {"DeepSeek-R1-Distill-Qwen-14B-Q4_0.gguf"},
      {"DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"},
      {"DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf"},
      {"DeepSeek-R1-Distill-Qwen-14B-f32/DeepSeek-R1-Distill-Qwen-14B-f32", 2}},
     {
      {"", "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"IQ2_M", "DeepSeek-R1-Distill-Qwen-14B-IQ2_M.gguf", 1, "", "", "", "", ""},
      {"F32", "DeepSeek-R1-Distill-Qwen-14B-f32/DeepSeek-R1-Distill-Qwen-14B-f32-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"F32", "Q8_0", "Q4_K_M", "Q4_0", "IQ2_M"}},

    {"bartowski/Qwen2.5-72B-Instruct-GGUF",
     {{"Qwen2.5-72B-Instruct-IQ1_M.gguf"},
      {"Qwen2.5-72B-Instruct-Q3_K_L.gguf"},
      {"Qwen2.5-72B-Instruct-Q4_K_M.gguf"},
      {"Qwen2.5-72B-Instruct-Q8_0/Qwen2.5-72B-Instruct-Q8_0", 2}},
     {
      {"", "Qwen2.5-72B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen2.5-72B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Qwen2.5-72B-Instruct-Q8_0/Qwen2.5-72B-Instruct-Q8_0-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"IQ1_M", "Qwen2.5-72B-Instruct-IQ1_M.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_L", "IQ1_M"}},

    {"batiai/gemma-4-E2B-it-GGUF",
     {{"google-gemma-4-E2B-it-Q4_K_M.gguf"},
      {"google-gemma-4-E2B-it-Q6_K.gguf"},
      {"mmproj-BF16.gguf"}},
     {
      {"", "google-gemma-4-E2B-it-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "google-gemma-4-E2B-it-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q6_K", "google-gemma-4-E2B-it-Q6_K.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q6_K", "Q4_K_M"}},

    {"batiai/Qwen3.6-27B-GGUF",
     {{"Qwen-Qwen3.6-27B-IQ3_XXS.gguf"},
      {"Qwen-Qwen3.6-27B-Q3_K_M.gguf"},
      {"Qwen-Qwen3.6-27B-Q4_K_M.gguf"},
      {"Qwen-Qwen3.6-27B-Q6_K.gguf"},
      {"mmproj-Qwen-Qwen3.6-27B-BF16.gguf"},
      {"mmproj-Qwen-Qwen3.6-27B-Q6_K.gguf"}},
     {
      {"", "Qwen-Qwen3.6-27B-Q4_K_M.gguf", 1, "mmproj-Qwen-Qwen3.6-27B-Q6_K.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen-Qwen3.6-27B-Q4_K_M.gguf", 1, "mmproj-Qwen-Qwen3.6-27B-Q6_K.gguf", "", "", "", ""},
      {"IQ3_XXS", "Qwen-Qwen3.6-27B-IQ3_XXS.gguf", 1, "mmproj-Qwen-Qwen3.6-27B-Q6_K.gguf", "", "", "", ""},
      {"Q6_K", "Qwen-Qwen3.6-27B-Q6_K.gguf", 1, "mmproj-Qwen-Qwen3.6-27B-Q6_K.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q6_K", "Q4_K_M", "Q3_K_M", "IQ3_XXS"}},

    {"BattleGhost/Qwen3.8-Flash-Next-UNCENSORED-IQ2_XXS-GGUF",
     {{"Qwen3.8-Flash-Next-UNCENSORED-IQ2_XXS.gguf"}},
     {
      {"", "Qwen3.8-Flash-Next-UNCENSORED-IQ2_XXS.gguf", 1, "", "", "", "", ""},
      {"IQ2_XXS", "Qwen3.8-Flash-Next-UNCENSORED-IQ2_XXS.gguf", 1, "", "", "", "", ""}},
     {"IQ2_XXS"}},

    {"befox/WAN2.2-14B-Rapid-AllInOne-GGUF",
     {{"Mega-v1/wan2.2-rapid-mega-aio-nsfw-v1-Q2_K.gguf"},
      {"Mega-v1/wan2.2-rapid-mega-aio-nsfw-v1-Q8_0.gguf"},
      {"Mega-v1/wan2.2-rapid-mega-aio-v1-Q2_K.gguf"},
      {"Mega-v1/wan2.2-rapid-mega-aio-v1-Q8_0.gguf"},
      {"Mega-v10/wan2.2-rapid-mega-aio-nsfw-v10-Q8_0.gguf"},
      {"Mega-v10/wan2.2-rapid-mega-aio-v10-Q8_0.gguf"},
      {"Mega-v12/wan2.2-rapid-mega-aio-nsfw-v12.1-Q8_0.gguf"},
      {"Mega-v12/wan2.2-rapid-mega-aio-v12-Q8_0.gguf"},
      {"Mega-v2/wan2.2-rapid-mega-aio-nsfw-v2-Q8_0.gguf"},
      {"Mega-v2/wan2.2-rapid-mega-aio-v2-Q8_0.gguf"},
      {"Mega-v3/wan2.2-rapid-mega-aio-v3-Q5_K.gguf"},
      {"Mega-v3/wan2.2-rapid-mega-aio-v3-Q8_0.gguf"},
      {"Mega-v3/wan2.2-rapid-mega-nsfw-aio-v3.1-Q5_K.gguf"},
      {"Mega-v3/wan2.2-rapid-mega-nsfw-aio-v3.1-Q8_0.gguf"},
      {"Mega-v4/wan2.2-rapid-mega-aio-v4-Q8_0.gguf"},
      {"Mega-v5/wan2.2-rapid-mega-aio-nsfw-v5-Q8_0.gguf"},
      {"Mega-v5/wan2.2-rapid-mega-aio-v5-Q8_0.gguf"},
      {"Mega-v7/wan2.2-rapid-mega-aio-v7-Q8_0.gguf"},
      {"v10/wan2.2-i2v-rapid-aio-v10-Q8_0.gguf"},
      {"v10/wan2.2-i2v-rapid-aio-v10-nsfw-Q8_0.gguf"},
      {"v10/wan2.2-t2v-rapid-aio-v10-Q8_0.gguf"},
      {"v10/wan2.2-t2v-rapid-aio-v10-nsfw-Q8_0.gguf"}},
     {
      {"", "Mega-v1/wan2.2-rapid-mega-aio-nsfw-v1-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Mega-v1/wan2.2-rapid-mega-aio-nsfw-v1-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "Mega-v1/wan2.2-rapid-mega-aio-nsfw-v1-Q2_K.gguf", 1, "", "", "", "", ""},
      {"Q5_K", "Mega-v3/wan2.2-rapid-mega-aio-v3-Q5_K.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_K", "Q2_K"}},

    {"Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF",
     {{"Qwen3.8-27B-ABLITERATED-Q2_K.gguf"},
      {"Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf"},
      {"Qwen3.8-27B-ABLITERATED-Q4_K_S.gguf"},
      {"Qwen3.8-27B-ABLITERATED-Q8_0.gguf"},
      {"dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf"},
      {"mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf"},
      {"mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf"}},
     {
      {"", "Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf", "", "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf", "", "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf", "", ""},
      {"Q8_0", "Qwen3.8-27B-ABLITERATED-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf", "", "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf", "", ""},
      {"Q2_K", "Qwen3.8-27B-ABLITERATED-Q2_K.gguf", 1, "mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf", "", "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf", "", ""},
      {"BF16", "", 0, "", "", "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf", "", ""}},
     {"Q8_0", "Q4_K_S", "Q4_K_M", "Q2_K"}},

    {"bloomer010/Ling-3.0-flash-GGUF",
     {{"Ling-3.0-flash-BF16.gguf"},
      {"Ling-3.0-flash-Q4_K_M.gguf"},
      {"Ling-3.0-flash-Q4_K_S.gguf"},
      {"Ling-3.0-flash-Q8_0.gguf"},
      {"Ling-3.0-flash-UD-Q8_K_XL.gguf"},
      {"Ling-3.0-flash-dspark-BF16.gguf"},
      {"Ling-3.0-flash-dspark-Q4_K_M.gguf"}},
     {
      {"", "Ling-3.0-flash-Q4_K_M.gguf", 1, "", "", "", "", "Ling-3.0-flash-dspark-Q4_K_M.gguf"},
      {"Q4_K_M", "Ling-3.0-flash-Q4_K_M.gguf", 1, "", "", "", "", "Ling-3.0-flash-dspark-Q4_K_M.gguf"},
      {"Q8_0", "Ling-3.0-flash-Q8_0.gguf", 1, "", "", "", "", "Ling-3.0-flash-dspark-Q4_K_M.gguf"},
      {"BF16", "Ling-3.0-flash-BF16.gguf", 1, "", "", "", "", "Ling-3.0-flash-dspark-BF16.gguf"},
      {"UD-Q8_K_XL", "Ling-3.0-flash-UD-Q8_K_XL.gguf", 1, "", "", "", "", "Ling-3.0-flash-dspark-Q4_K_M.gguf"}},
     {"Q8_K_XL", "Q8_0", "Q4_K_S", "Q4_K_M", "BF16"}},

    {"BnhHo/Qwen3.8-27B-Uncensored-GGUF",
     {{"Qwen3.8-27B-Uncensored-IQ2_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q5_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q8_0.gguf"},
      {"Qwen3.8-27B-Uncensored-draft-Q4_0.gguf"},
      {"Qwen3.8-27B-Uncensored-draft-Q8_0.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-Q5_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf"},
      {"mmproj-Qwen3.8-27B-Uncensored-F16.gguf"}},
     {
      {"", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.8-27B-Uncensored-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"IQ2_M", "Qwen3.8-27B-Uncensored-IQ2_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"Q4_0", "Qwen3.8-27B-Uncensored-draft-Q4_0.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "IQ2_M", "Q4_0"}},

    {"bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF",
     {{"ThinkingCap-Qwen3.8-27B-IQ4_XS.gguf"},
      {"ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf"},
      {"ThinkingCap-Qwen3.8-27B-Q6_K.gguf"},
      {"ThinkingCap-Qwen3.8-27B-Q8_0.gguf"},
      {"ThinkingCap-Qwen3.8-27B-f16.gguf"},
      {"mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf"}},
     {
      {"", "ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf", 1, "mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf", 1, "mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf", "", "", "", ""},
      {"Q8_0", "ThinkingCap-Qwen3.8-27B-Q8_0.gguf", 1, "mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf", "", "", "", ""},
      {"IQ4_XS", "ThinkingCap-Qwen3.8-27B-IQ4_XS.gguf", 1, "mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf", "", "", "", ""},
      {"F16", "ThinkingCap-Qwen3.8-27B-f16.gguf", 1, "mmproj-ThinkingCap-Qwen3.8-27B-f16.gguf", "", "", "", ""}},
     {"F16", "Q8_0", "Q6_K", "Q4_K_M", "IQ4_XS"}},

    {"bytkim/Qwen3.8-27B-pi-GGUF",
     {{"Qwen3.8-27B-pi-BF16.gguf"},
      {"Qwen3.8-27B-pi-Q3_K_M.gguf"},
      {"Qwen3.8-27B-pi-Q4_K_M.gguf"},
      {"Qwen3.8-27B-pi-Q8_0.gguf"},
      {"dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"},
      {"dflash2/Qwen3.8-27B-DFlash2-Q8_0.gguf"},
      {"mmproj-Qwen3.8-27B-pi-BF16.gguf"},
      {"mtp/mtp-Qwen3.8-27B-pi-BF16.gguf"},
      {"mtp/mtp-Qwen3.8-27B-pi-Q4_0.gguf"},
      {"mtp/mtp-Qwen3.8-27B-pi-Q8_0.gguf"}},
     {
      {"", "Qwen3.8-27B-pi-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-pi-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-pi-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-pi-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.8-27B-pi-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-pi-BF16.gguf", "", "", "", ""},
      {"BF16", "Qwen3.8-27B-pi-BF16.gguf", 1, "mmproj-Qwen3.8-27B-pi-BF16.gguf", "", "", "", ""},
      {"Q4_0", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_M", "BF16"}},

    {"CaptainArni/audio.cpp-gguf",
     {{"ACE-Step1.5-GGUF/xl-sft-q8dit/ace-step-1.5-xl-sft-q8dit.gguf"},
      {"ACE-Step1.5-GGUF/xl-turbo-q8dit/ace-step-1.5-xl-turbo-q8dit.gguf"},
      {"ACE-Step1.5-GGUF/xl-turbo/ace-step-1.5-xl-turbo-bf16.gguf"}},
     {
      {"", "ACE-Step1.5-GGUF/xl-sft-q8dit/ace-step-1.5-xl-sft-q8dit.gguf", 1, "", "", "", "", ""},
      {"Q8DIT", "ACE-Step1.5-GGUF/xl-sft-q8dit/ace-step-1.5-xl-sft-q8dit.gguf", 1, "", "", "", "", ""},
      {"BF16", "ACE-Step1.5-GGUF/xl-turbo/ace-step-1.5-xl-turbo-bf16.gguf", 1, "", "", "", "", ""}},
     {"BF16", "Q8DIT"}},

    {"cesarsal1nas/Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-Q4_K_M-GGUF",
     {{"Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-Q4_K_M.gguf"},
      {"Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-mmproj-F16.gguf"}},
     {
      {"", "Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-Q4_K_M.gguf", 1, "Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-mmproj-F16.gguf", "", "", "", ""},
      {"Q4_K_M", "Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-Q4_K_M.gguf", 1, "Huihui-Qwen3.5-35B-A3B-Claude-4.6-Opus-abliterated-mmproj-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"chimingw/Qwen3.8-27B-Uncensored-OrcaRouter-GGUF",
     {{"AUX/mmproj-Qwen3.8-27B-Uncensored-OrcaRouter-BF16.gguf"},
      {"AUX/mmproj-Qwen3.8-27B-Uncensored-OrcaRouter-F16.gguf"},
      {"BF16/Qwen3.8-27B-Uncensored-OrcaRouter-FP8-derived-BF16", 2},
      {"MTP/Qwen3.8-27B-Uncensored-OrcaRouter-MTP-Q8_0.gguf"},
      {"Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-OrcaRouter-Q5_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-OrcaRouter-Q8_0.gguf"}},
     {
      {"", "Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-Uncensored-OrcaRouter-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "MTP/Qwen3.8-27B-Uncensored-OrcaRouter-MTP-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "BF16/Qwen3.8-27B-Uncensored-OrcaRouter-FP8-derived-BF16-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"Q5_K_M", "Qwen3.8-27B-Uncensored-OrcaRouter-Q5_K_M.gguf", 1, "", "", "", "", ""},
      {"F16", "BF16/Qwen3.8-27B-Uncensored-OrcaRouter-FP8-derived-BF16-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "BF16"}},

    {"ChrisColeTech/qwen-image-edit-uncensored-GGUF",
     {{"split/diffusion_models/qwen-image-edit-2511-uncensored-Q4_K_M.gguf"},
      {"split/diffusion_models/qwen-image-edit-2511-uncensored-Q8_0.gguf"},
      {"split/text_encoders/Qwen2.5-VL-7B-Instruct-q4_0.gguf"}},
     {
      {"", "split/diffusion_models/qwen-image-edit-2511-uncensored-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "split/diffusion_models/qwen-image-edit-2511-uncensored-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "split/diffusion_models/qwen-image-edit-2511-uncensored-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q4_0", "split/text_encoders/Qwen2.5-VL-7B-Instruct-q4_0.gguf", 1, "", "", "", "", ""}},
     {"Q4_0", "Q8_0", "Q4_K_M"}},

    {"city96/FLUX.1-dev-gguf",
     {{"flux1-dev-F16.gguf"},
      {"flux1-dev-Q4_K_S.gguf"},
      {"flux1-dev-Q8_0.gguf"}},
     {
      {"", "flux1-dev-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "flux1-dev-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "flux1-dev-F16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_S", "F16"}},

    {"culturerevolt/gemma-4-12b-heretic-abliterated-GGUF",
     {{"gemma-4-12b-heretic-IQ3_XS.gguf"},
      {"gemma-4-12b-heretic-Q4_K_M.gguf"},
      {"gemma-4-12b-heretic-Q5_K_M.gguf"},
      {"gemma-4-12b-heretic-abliterated-Q8_0.gguf"},
      {"gemma-4-12b-heretic-mmproj-f16.gguf"}},
     {
      {"", "gemma-4-12b-heretic-Q4_K_M.gguf", 1, "gemma-4-12b-heretic-mmproj-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "gemma-4-12b-heretic-Q4_K_M.gguf", 1, "gemma-4-12b-heretic-mmproj-f16.gguf", "", "", "", ""},
      {"Q8_0", "gemma-4-12b-heretic-abliterated-Q8_0.gguf", 1, "gemma-4-12b-heretic-mmproj-f16.gguf", "", "", "", ""},
      {"IQ3_XS", "gemma-4-12b-heretic-IQ3_XS.gguf", 1, "gemma-4-12b-heretic-mmproj-f16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "IQ3_XS"}},

    {"CypherTroopers/CPH-Community-7B",
     {{"cph-community-7b-q4_k_m.gguf"}},
     {
      {"", "cph-community-7b-q4_k_m.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "cph-community-7b-q4_k_m.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"DamRsn/muscriptor-gguf",
     {{"v1/muscriptor-large-f16.gguf"},
      {"v1/muscriptor-medium-f16.gguf"},
      {"v1/muscriptor-small-f16.gguf"}},
     {
      {"", "v1/muscriptor-large-f16.gguf", 1, "", "", "", "", ""},
      {"F16", "v1/muscriptor-large-f16.gguf", 1, "", "", "", "", ""}},
     {"F16"}},

    {"DavidAU/LFM2.5-2.6B-Qwen3.8-Turbo-Brilliance-Power-X12-NEO-MAX-GGUF",
     {{"LFM2.5-2.6B-Q3.8-TBrilliance-BF16.gguf"},
      {"LFM2.5-2.6B-Q3.8-TBrilliance-NEO-MAX-Q6_K.gguf"},
      {"LFM2.5-2.6B-Q3.8-TBrilliance-NEO-MAX-Q8_0.gguf"},
      {"LFM2.5-2.6B-Q3.8-TBrilliance-NEO-Q6_K.gguf"},
      {"LFM2.5-2.6B-Q3.8-TBrilliance-NEO-Q8_0.gguf"}},
     {
      {"", "LFM2.5-2.6B-Q3.8-TBrilliance-NEO-MAX-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "LFM2.5-2.6B-Q3.8-TBrilliance-NEO-MAX-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "LFM2.5-2.6B-Q3.8-TBrilliance-BF16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q6_K", "BF16"}},

    {"DavidAU/OpenAi-GPT-oss-20b-HERETIC-uncensored-NEO-Imatrix-gguf",
     {{"OpenAI-20B-NEO-CODE-DI-Uncensored-Q5_1.gguf"},
      {"OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf"},
      {"OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf"},
      {"OpenAI-20B-NEO-CODEPlus-Uncensored-Q5_1.gguf"},
      {"OpenAI-20B-NEO-CODEPlus16-Uncensored-IQ4_NL.gguf"},
      {"OpenAI-20B-NEO-HRR-CODE-5-TRI-Uncensored-Q8_0.gguf"},
      {"OpenAI-20B-NEO-HRR-CODE-TRI-Uncensored-IQ4_NL.gguf"},
      {"OpenAI-20B-NEO-HRR-CODE-TRI-Uncensored-Q5_1.gguf"},
      {"OpenAI-20B-NEO-HRR-CODE-TRI-Uncensored-Q8_0.gguf"},
      {"OpenAI-20B-NEO-HRR-DI-Uncensored-Q5_1.gguf"},
      {"OpenAI-20B-NEO-HRR-DI-Uncensored-Q8_0.gguf"},
      {"OpenAI-20B-NEO-HRRPlus-Uncensored-IQ4_NL.gguf"},
      {"OpenAI-20B-NEO-Uncensored2-IQ4_NL.gguf"},
      {"OpenAI-20B-NEO-Uncensored2-Q5_1.gguf"},
      {"OpenAI-20B-NEOPlus-Uncensored-IQ4_NL.gguf"},
      {"OpenAI-20B-NEOPlus-Uncensored-Q5_1.gguf"},
      {"OpenAI-20B-NEOPlus-Uncensored-Q8_0.gguf"}},
     {
      {"", "OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "OpenAI-20B-NEO-CODE-DI-Uncensored-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q5_1", "OpenAI-20B-NEO-CODE-DI-Uncensored-Q5_1.gguf", 1, "", "", "", "", ""},
      {"IQ4_NL", "OpenAI-20B-NEO-CODEPlus-Uncensored-IQ4_NL.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_1", "IQ4_NL"}},

    {"DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF",
     {{"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-IQ2_M.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_S.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q8_0.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q4_K_M.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q4_K_S.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q8_0.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-NEO-MAX-MTP-Q8_0.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-NEO-MAX-MTP-bf16.gguf"},
      {"Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-TOOLS-NEO-MAX-MTP-Q8_0.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-MTP-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"IQ2_M", "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-IQ2_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-NEO-MAX-MTP-bf16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-plusIQ-NEO-MAX-MTP-bf16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_0", "BF16", "Q4_K_S", "Q4_K_M", "IQ2_M"}},

    {"DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF",
     {{"Qwen3.6-27B-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf"},
      {"Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf"},
      {"Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q6_K.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-HIGH-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q6_K", "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q6_K.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q6_K", "Q4_K_M", "Q8_0"}},

    {"dcostenco/smollm2-360m-aac-watch",
     {{"smollm2-360m-aac-q3ks.gguf"}},
     {
      {"", "smollm2-360m-aac-q3ks.gguf", 1, "", "", "", "", ""},
      {"Q3KS", "smollm2-360m-aac-q3ks.gguf", 1, "", "", "", "", ""}},
     {"Q3KS"}},

    {"DeepBeepMeep/LTX-2",
     {{"ltx-2.3-22b-distilled-Q4_K_M_light.gguf"},
      {"ltx-2.3-22b-distilled-Q6_K_light.gguf"},
      {"ltx-2.3-22b-distilled-Q8_0_light.gguf"}},
     {
      {"", "ltx-2.3-22b-distilled-Q4_K_M_light.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M_LIGHT", "ltx-2.3-22b-distilled-Q4_K_M_light.gguf", 1, "", "", "", "", ""},
      {"Q8_0_LIGHT", "ltx-2.3-22b-distilled-Q8_0_light.gguf", 1, "", "", "", "", ""}},
     {"Q8_0_LIGHT", "Q6_K_LIGHT", "Q4_K_M_LIGHT"}},

    {"defog/sqlcoder-7b-2",
     {{"sqlcoder-7b-q5_k_m.gguf"}},
     {
      {"", "sqlcoder-7b-q5_k_m.gguf", 1, "", "", "", "", ""},
      {"Q5_K_M", "sqlcoder-7b-q5_k_m.gguf", 1, "", "", "", "", ""}},
     {"Q5_K_M"}},

    {"dgflash/FLUX.1-dev-4bit-gguf",
     {{"flux1-dev-Q4_K_S.gguf"}},
     {
      {"", "flux1-dev-Q4_K_S.gguf", 1, "", "", "", "", ""},
      {"Q4_K_S", "flux1-dev-Q4_K_S.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_S"}},

    {"empero-ai/Qwythos-27B-v1-GGUF",
     {{"Qwythos-27B-BF16.gguf"},
      {"Qwythos-27B-MTP-BF16.gguf"},
      {"Qwythos-27B-MTP-Q4_K_M.gguf"},
      {"Qwythos-27B-MTP-Q5_K_M.gguf"},
      {"Qwythos-27B-MTP-Q8_0.gguf"},
      {"Qwythos-27B-Q4_K_M.gguf"},
      {"Qwythos-27B-Q5_K_M.gguf"},
      {"Qwythos-27B-Q8_0.gguf"},
      {"mmproj-Qwythos-27B-F16.gguf"}},
     {
      {"", "Qwythos-27B-MTP-Q4_K_M.gguf", 1, "mmproj-Qwythos-27B-F16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwythos-27B-MTP-Q4_K_M.gguf", 1, "mmproj-Qwythos-27B-F16.gguf", "", "", "", ""},
      {"Q8_0", "Qwythos-27B-MTP-Q8_0.gguf", 1, "mmproj-Qwythos-27B-F16.gguf", "", "", "", ""},
      {"BF16", "Qwythos-27B-BF16.gguf", 1, "mmproj-Qwythos-27B-F16.gguf", "", "", "", ""},
      {"F16", "Qwythos-27B-BF16.gguf", 1, "mmproj-Qwythos-27B-F16.gguf", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "BF16"}},

    {"empero-ai/Qwythos-9B-v2-GGUF",
     {{"Qwythos-9B-v2-BF16.gguf"},
      {"Qwythos-9B-v2-MTP-BF16.gguf"},
      {"Qwythos-9B-v2-MTP-Q4_K_M.gguf"},
      {"Qwythos-9B-v2-MTP-Q5_K_M.gguf"},
      {"Qwythos-9B-v2-MTP-Q8_0.gguf"},
      {"Qwythos-9B-v2-Q4_K_M.gguf"},
      {"Qwythos-9B-v2-Q5_K_M.gguf"},
      {"Qwythos-9B-v2-Q8_0.gguf"},
      {"mmproj-Qwythos-9B-v2-BF16.gguf"}},
     {
      {"", "Qwythos-9B-v2-MTP-Q4_K_M.gguf", 1, "mmproj-Qwythos-9B-v2-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwythos-9B-v2-MTP-Q4_K_M.gguf", 1, "mmproj-Qwythos-9B-v2-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwythos-9B-v2-MTP-Q8_0.gguf", 1, "mmproj-Qwythos-9B-v2-BF16.gguf", "", "", "", ""},
      {"BF16", "Qwythos-9B-v2-BF16.gguf", 1, "mmproj-Qwythos-9B-v2-BF16.gguf", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "BF16"}},

    {"FINAL-Bench/POCKET-35B-GGUF",
     {{"POCKET-35B-IQ1_M.gguf"},
      {"POCKET-35B-Q3_K_M.gguf"},
      {"POCKET-35B-Q4_K_M.gguf"}},
     {
      {"", "POCKET-35B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "POCKET-35B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"IQ1_M", "POCKET-35B-IQ1_M.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M", "Q3_K_M", "IQ1_M"}},

    {"FINAL-Bench/POCKET-Darwin-180B-GGUF",
     {{"POCKET-Darwin-180B-UD-Q4_K_XL", 4}},
     {
      {"", "POCKET-Darwin-180B-UD-Q4_K_XL-00001-of-00004.gguf", 4, "", "", "", "", ""},
      {"UD-Q4_K_XL", "POCKET-Darwin-180B-UD-Q4_K_XL-00001-of-00004.gguf", 4, "", "", "", "", ""}},
     {"Q4_K_XL"}},

    {"FunAudioLLM/Fun-ASR-Nano-2512-GGUF",
     {{"fun-asr-nano-2512-f16.gguf"},
      {"fun-asr-nano-2512-q8_0.gguf"}},
     {
      {"", "fun-asr-nano-2512-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "fun-asr-nano-2512-q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "fun-asr-nano-2512-f16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "F16"}},

    {"GCSA-AiLab/GLM-5.3-Flash-Uncensored-RCO-GSQ-GGUF",
     {{"Q4/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4.gguf"},
      {"Q8/GLM-5.3-Flash-Uncensored-Q8_0.gguf"}},
     {
      {"", "Q8/GLM-5.3-Flash-Uncensored-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Q8/GLM-5.3-Flash-Uncensored-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q4", "Q4/GLM-5.3-Flash-Uncensored-GSQ-RCO-Q4.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4"}},

    {"ggml-org/embeddinggemma-2-GGUF",
     {{"embeddinggemma-2-BF16.gguf"},
      {"embeddinggemma-2-Q8_0.gguf"},
      {"mmproj-embeddinggemma-2-BF16.gguf"},
      {"mmproj-embeddinggemma-2-Q8_0.gguf"}},
     {
      {"", "embeddinggemma-2-Q8_0.gguf", 1, "mmproj-embeddinggemma-2-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "embeddinggemma-2-Q8_0.gguf", 1, "mmproj-embeddinggemma-2-Q8_0.gguf", "", "", "", ""},
      {"BF16", "embeddinggemma-2-BF16.gguf", 1, "mmproj-embeddinggemma-2-BF16.gguf", "", "", "", ""}},
     {"Q8_0", "BF16"}},

    {"ggml-org/gemma-3-4b-it-GGUF",
     {{"gemma-3-4b-it-Q4_K_M.gguf"},
      {"gemma-3-4b-it-Q8_0.gguf"},
      {"gemma-3-4b-it-f16.gguf"},
      {"mmproj-model-f16.gguf"}},
     {
      {"", "gemma-3-4b-it-Q4_K_M.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "gemma-3-4b-it-Q4_K_M.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"Q8_0", "gemma-3-4b-it-Q8_0.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""},
      {"F16", "gemma-3-4b-it-f16.gguf", 1, "mmproj-model-f16.gguf", "", "", "", ""}},
     {"F16", "Q8_0", "Q4_K_M"}},

    {"ggml-org/gemma-4-12B-it-GGUF",
     {{"gemma-4-12B-it-BF16.gguf"},
      {"gemma-4-12B-it-Q4_0.gguf"},
      {"gemma-4-12B-it-Q8_0.gguf"},
      {"mmproj-gemma-4-12B-it-BF16.gguf"},
      {"mmproj-gemma-4-12B-it-Q8_0.gguf"},
      {"mtp-gemma-4-12B-it-BF16.gguf"},
      {"mtp-gemma-4-12B-it-Q4_0.gguf"},
      {"mtp-gemma-4-12B-it-Q8_0.gguf"}},
     {
      {"", "gemma-4-12B-it-Q8_0.gguf", 1, "mmproj-gemma-4-12B-it-Q8_0.gguf", "mtp-gemma-4-12B-it-Q8_0.gguf", "", "", ""},
      {"Q8_0", "gemma-4-12B-it-Q8_0.gguf", 1, "mmproj-gemma-4-12B-it-Q8_0.gguf", "mtp-gemma-4-12B-it-Q8_0.gguf", "", "", ""},
      {"BF16", "gemma-4-12B-it-BF16.gguf", 1, "mmproj-gemma-4-12B-it-BF16.gguf", "mtp-gemma-4-12B-it-BF16.gguf", "", "", ""}},
     {"Q8_0", "Q4_0", "BF16"}},

    {"ggml-org/gemma-4-26B-A4B-it-GGUF",
     {{"dflash-gemma-4-26B-A4B-it-BF16.gguf"},
      {"dflash-gemma-4-26B-A4B-it-Q8_0.gguf"},
      {"gemma-4-26B-A4B-it-BF16.gguf"},
      {"gemma-4-26B-A4B-it-Q4_0.gguf"},
      {"gemma-4-26B-A4B-it-Q8_0.gguf"},
      {"mmproj-gemma-4-26B-A4B-it-BF16.gguf"},
      {"mmproj-gemma-4-26B-A4B-it-Q8_0.gguf"},
      {"mtp-gemma-4-26B-A4B-it-BF16.gguf"},
      {"mtp-gemma-4-26B-A4B-it-Q4_0.gguf"},
      {"mtp-gemma-4-26B-A4B-it-Q8_0.gguf"}},
     {
      {"", "gemma-4-26B-A4B-it-Q8_0.gguf", 1, "mmproj-gemma-4-26B-A4B-it-Q8_0.gguf", "mtp-gemma-4-26B-A4B-it-Q8_0.gguf", "dflash-gemma-4-26B-A4B-it-Q8_0.gguf", "", ""},
      {"Q8_0", "gemma-4-26B-A4B-it-Q8_0.gguf", 1, "mmproj-gemma-4-26B-A4B-it-Q8_0.gguf", "mtp-gemma-4-26B-A4B-it-Q8_0.gguf", "dflash-gemma-4-26B-A4B-it-Q8_0.gguf", "", ""},
      {"BF16", "gemma-4-26B-A4B-it-BF16.gguf", 1, "mmproj-gemma-4-26B-A4B-it-BF16.gguf", "mtp-gemma-4-26B-A4B-it-BF16.gguf", "dflash-gemma-4-26B-A4B-it-BF16.gguf", "", ""}},
     {"Q8_0", "Q4_0", "BF16"}},

    {"ggml-org/gemma-4-E2B-it-GGUF",
     {{"gemma-4-E2B-it-BF16.gguf"},
      {"gemma-4-E2B-it-Q4_0.gguf"},
      {"gemma-4-E2B-it-Q8_0.gguf"},
      {"mmproj-gemma-4-E2B-it-BF16.gguf"},
      {"mmproj-gemma-4-E2B-it-Q8_0.gguf"},
      {"mtp-gemma-4-E2B-it-BF16.gguf"},
      {"mtp-gemma-4-E2B-it-Q4_0.gguf"},
      {"mtp-gemma-4-E2B-it-Q8_0.gguf"}},
     {
      {"", "gemma-4-E2B-it-Q8_0.gguf", 1, "mmproj-gemma-4-E2B-it-Q8_0.gguf", "mtp-gemma-4-E2B-it-Q8_0.gguf", "", "", ""},
      {"Q8_0", "gemma-4-E2B-it-Q8_0.gguf", 1, "mmproj-gemma-4-E2B-it-Q8_0.gguf", "mtp-gemma-4-E2B-it-Q8_0.gguf", "", "", ""},
      {"BF16", "gemma-4-E2B-it-BF16.gguf", 1, "mmproj-gemma-4-E2B-it-BF16.gguf", "mtp-gemma-4-E2B-it-BF16.gguf", "", "", ""}},
     {"Q8_0", "Q4_0", "BF16"}},

    {"ggml-org/GLM-4.7-Flash-GGUF",
     {{"GLM-4.7-Flash-Q4_K.gguf"},
      {"GLM-4.7-Flash-Q8_0.gguf"}},
     {
      {"", "GLM-4.7-Flash-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "GLM-4.7-Flash-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q4_K", "GLM-4.7-Flash-Q4_K.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K"}},

    {"ggml-org/GLM-OCR-GGUF",
     {{"GLM-OCR-Q8_0.gguf"},
      {"GLM-OCR-f16.gguf"},
      {"mmproj-GLM-OCR-Q8_0.gguf"}},
     {
      {"", "GLM-OCR-Q8_0.gguf", 1, "mmproj-GLM-OCR-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "GLM-OCR-Q8_0.gguf", 1, "mmproj-GLM-OCR-Q8_0.gguf", "", "", "", ""},
      {"F16", "GLM-OCR-f16.gguf", 1, "mmproj-GLM-OCR-Q8_0.gguf", "", "", "", ""}},
     {"F16", "Q8_0"}},

    {"ggml-org/gpt-oss-120b-GGUF",
     {{"eagle3-gpt-oss-120b-BF16.gguf"},
      {"eagle3-gpt-oss-120b-Q8_0.gguf"},
      {"gpt-oss-120b-MXFP4.gguf"}},
     {
      {"", "gpt-oss-120b-MXFP4.gguf", 1, "", "", "", "eagle3-gpt-oss-120b-Q8_0.gguf", ""},
      {"Q8_0", "", 0, "", "", "", "eagle3-gpt-oss-120b-Q8_0.gguf", ""},
      {"MXFP4", "gpt-oss-120b-MXFP4.gguf", 1, "", "", "", "eagle3-gpt-oss-120b-Q8_0.gguf", ""},
      {"BF16", "", 0, "", "", "", "eagle3-gpt-oss-120b-BF16.gguf", ""}},
     {"MXFP4"}},

    {"ggml-org/MiMo-V2.6-Distill-Qwen-9B-GGUF",
     {{"MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf"},
      {"mmproj-MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf"}},
     {
      {"", "MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf", 1, "mmproj-MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf", 1, "mmproj-MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf", "", "", "", ""}},
     {"Q8_0"}},

    {"ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
     {{"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf"},
      {"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"},
      {"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"},
      {"mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf"},
      {"mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf"},
      {"mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"}},
     {
      {"", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf", 1, "", "mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf", "", "", ""},
      {"Q8_0", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf", 1, "", "mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf", "", "", ""},
      {"BF16", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf", 1, "", "mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf", "", "", ""}},
     {"Q8_0", "Q4_0", "BF16"}},

    {"ggml-org/Qwen2.5-Omni-3B-GGUF",
     {{"Qwen2.5-Omni-3B-Q4_K_M.gguf"},
      {"Qwen2.5-Omni-3B-Q8_0.gguf"},
      {"Qwen2.5-Omni-3B-f16.gguf"},
      {"mmproj-Qwen2.5-Omni-3B-Q8_0.gguf"},
      {"mmproj-Qwen2.5-Omni-3B-f16.gguf"}},
     {
      {"", "Qwen2.5-Omni-3B-Q4_K_M.gguf", 1, "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen2.5-Omni-3B-Q4_K_M.gguf", 1, "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "Qwen2.5-Omni-3B-Q8_0.gguf", 1, "mmproj-Qwen2.5-Omni-3B-Q8_0.gguf", "", "", "", ""},
      {"F16", "Qwen2.5-Omni-3B-f16.gguf", 1, "mmproj-Qwen2.5-Omni-3B-f16.gguf", "", "", "", ""}},
     {"F16", "Q8_0", "Q4_K_M"}},

    {"ggml-org/Qwen2.5-VL-32B-Instruct-GGUF",
     {{"Qwen2.5-VL-32B-Instruct-Q4_K_M.gguf"},
      {"Qwen2.5-VL-32B-Instruct-Q8_0.gguf"},
      {"Qwen2.5-VL-32B-Instruct-f16", 2},
      {"mmproj-Qwen2.5-VL-32B-Instruct-Q8_0.gguf"},
      {"mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf"}},
     {
      {"", "Qwen2.5-VL-32B-Instruct-Q4_K_M.gguf", 1, "mmproj-Qwen2.5-VL-32B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen2.5-VL-32B-Instruct-Q4_K_M.gguf", 1, "mmproj-Qwen2.5-VL-32B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "Qwen2.5-VL-32B-Instruct-Q8_0.gguf", 1, "mmproj-Qwen2.5-VL-32B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"F16", "Qwen2.5-VL-32B-Instruct-f16-00001-of-00002.gguf", 2, "mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf", "", "", "", ""}},
     {"F16", "Q8_0", "Q4_K_M"}},

    {"ggml-org/Qwen3-1.7B-GGUF",
     {{"Qwen3-1.7B-Q4_K_M.gguf"},
      {"Qwen3-1.7B-Q8_0.gguf"},
      {"Qwen3-1.7B-f16.gguf"}},
     {
      {"", "Qwen3-1.7B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen3-1.7B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Qwen3-1.7B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "Qwen3-1.7B-f16.gguf", 1, "", "", "", "", ""}},
     {"F16", "Q8_0", "Q4_K_M"}},

    {"ggml-org/Qwen3-ASR-1.7B-GGUF",
     {{"Qwen3-ASR-1.7B-Q8_0.gguf"},
      {"Qwen3-ASR-1.7B-bf16.gguf"},
      {"mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"},
      {"mmproj-Qwen3-ASR-1.7B-bf16.gguf"}},
     {
      {"", "Qwen3-ASR-1.7B-Q8_0.gguf", 1, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3-ASR-1.7B-Q8_0.gguf", 1, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", "", "", "", ""},
      {"BF16", "Qwen3-ASR-1.7B-bf16.gguf", 1, "mmproj-Qwen3-ASR-1.7B-bf16.gguf", "", "", "", ""}},
     {"BF16", "Q8_0"}},

    {"ggml-org/Qwen3.6-27B-GGUF",
     {{"Qwen3.6-27B-BF16.gguf"},
      {"Qwen3.6-27B-Q4_K_M.gguf"},
      {"Qwen3.6-27B-Q8_0.gguf"},
      {"dflash-Qwen3.6-27B-BF16.gguf"},
      {"dflash-Qwen3.6-27B-Q8_0.gguf"},
      {"mmproj-Qwen3.6-27B-BF16.gguf"},
      {"mmproj-Qwen3.6-27B-Q8_0.gguf"},
      {"mtp-Qwen3.6-27B-BF16.gguf"},
      {"mtp-Qwen3.6-27B-Q4_0.gguf"},
      {"mtp-Qwen3.6-27B-Q8_0.gguf"}},
     {
      {"", "Qwen3.6-27B-Q4_K_M.gguf", 1, "mmproj-Qwen3.6-27B-Q8_0.gguf", "mtp-Qwen3.6-27B-Q4_0.gguf", "dflash-Qwen3.6-27B-Q8_0.gguf", "", ""},
      {"Q4_K_M", "Qwen3.6-27B-Q4_K_M.gguf", 1, "mmproj-Qwen3.6-27B-Q8_0.gguf", "mtp-Qwen3.6-27B-Q4_0.gguf", "dflash-Qwen3.6-27B-Q8_0.gguf", "", ""},
      {"Q8_0", "Qwen3.6-27B-Q8_0.gguf", 1, "mmproj-Qwen3.6-27B-Q8_0.gguf", "mtp-Qwen3.6-27B-Q8_0.gguf", "dflash-Qwen3.6-27B-Q8_0.gguf", "", ""},
      {"BF16", "Qwen3.6-27B-BF16.gguf", 1, "mmproj-Qwen3.6-27B-BF16.gguf", "mtp-Qwen3.6-27B-BF16.gguf", "dflash-Qwen3.6-27B-BF16.gguf", "", ""},
      {"Q4_0", "", 0, "", "mtp-Qwen3.6-27B-Q4_0.gguf", "dflash-Qwen3.6-27B-Q8_0.gguf", "", ""}},
     {"Q8_0", "Q4_K_M", "BF16"}},

    {"ggml-org/Qwen3.8-27B-GGUF",
     {{"Qwen3.8-27B-BF16.gguf"},
      {"Qwen3.8-27B-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Q8_0.gguf"},
      {"dflash-Qwen3.8-27B-BF16.gguf"},
      {"dflash-Qwen3.8-27B-Q4_0.gguf"},
      {"dflash-Qwen3.8-27B-Q8_0.gguf"},
      {"mmproj-Qwen3.8-27B-BF16.gguf"},
      {"mmproj-Qwen3.8-27B-Q8_0.gguf"},
      {"mtp-Qwen3.8-27B-BF16.gguf"},
      {"mtp-Qwen3.8-27B-Q4_0.gguf"},
      {"mtp-Qwen3.8-27B-Q8_0.gguf"}},
     {
      {"", "Qwen3.8-27B-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Q8_0.gguf", "mtp-Qwen3.8-27B-Q4_0.gguf", "dflash-Qwen3.8-27B-Q4_0.gguf", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Q8_0.gguf", "mtp-Qwen3.8-27B-Q4_0.gguf", "dflash-Qwen3.8-27B-Q4_0.gguf", "", ""},
      {"Q8_0", "Qwen3.8-27B-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-Q8_0.gguf", "mtp-Qwen3.8-27B-Q8_0.gguf", "dflash-Qwen3.8-27B-Q8_0.gguf", "", ""},
      {"BF16", "Qwen3.8-27B-BF16.gguf", 1, "mmproj-Qwen3.8-27B-BF16.gguf", "mtp-Qwen3.8-27B-BF16.gguf", "dflash-Qwen3.8-27B-BF16.gguf", "", ""},
      {"Q4_0", "", 0, "", "mtp-Qwen3.8-27B-Q4_0.gguf", "dflash-Qwen3.8-27B-Q4_0.gguf", "", ""}},
     {"Q8_0", "Q4_K_M", "BF16"}},

    {"ggml-org/Qwen3.8-Flash-Next-GGUF",
     {{"Qwen3.8-Flash-Next-IQ4_NL", 2},
      {"Qwen3.8-Flash-Next-Q2_K", 2},
      {"Qwen3.8-Flash-Next-Q8_0", 2},
      {"mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"},
      {"mtp-Qwen3.8-Flash-Next-BF16.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q4_0.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q8_0.gguf"}},
     {
      {"", "Qwen3.8-Flash-Next-Q8_0-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"Q8_0", "Qwen3.8-Flash-Next-Q8_0-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"IQ4_NL", "Qwen3.8-Flash-Next-IQ4_NL-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-Q8_0.gguf", "mtp-Qwen3.8-Flash-Next-Q4_0.gguf", "", "", ""},
      {"BF16", "", 0, "", "mtp-Qwen3.8-Flash-Next-BF16.gguf", "", "", ""}},
     {"Q8_0", "Q2_K", "IQ4_NL"}},

    {"ggml-org/SmolVLM-500M-Instruct-GGUF",
     {{"SmolVLM-500M-Instruct-Q8_0.gguf"},
      {"SmolVLM-500M-Instruct-f16.gguf"},
      {"mmproj-SmolVLM-500M-Instruct-Q8_0.gguf"},
      {"mmproj-SmolVLM-500M-Instruct-f16.gguf"}},
     {
      {"", "SmolVLM-500M-Instruct-Q8_0.gguf", 1, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "SmolVLM-500M-Instruct-Q8_0.gguf", 1, "mmproj-SmolVLM-500M-Instruct-Q8_0.gguf", "", "", "", ""},
      {"F16", "SmolVLM-500M-Instruct-f16.gguf", 1, "mmproj-SmolVLM-500M-Instruct-f16.gguf", "", "", "", ""}},
     {"F16", "Q8_0"}},

    {"ggml-org/test-model-stories260K",
     {{"stories260K-f32.gguf"}},
     {
      {"", "stories260K-f32.gguf", 1, "", "", "", "", ""},
      {"F32", "stories260K-f32.gguf", 1, "", "", "", "", ""}},
     {"F32"}},

    {"GnLOLot/MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-GGUF",
     {{"MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-F16.gguf"},
      {"MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-Q8_0.gguf"}},
     {
      {"", "MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-F16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "F16"}},

    {"google/gemma-4-12B-it-qat-q4_0-gguf",
     {{"gemma-4-12b-it-qat-q4_0.gguf"},
      {"mmproj-gemma-4-12b-it-qat-q4_0.gguf"}},
     {
      {"", "gemma-4-12b-it-qat-q4_0.gguf", 1, "mmproj-gemma-4-12b-it-qat-q4_0.gguf", "", "", "", ""},
      {"Q4_0", "gemma-4-12b-it-qat-q4_0.gguf", 1, "mmproj-gemma-4-12b-it-qat-q4_0.gguf", "", "", "", ""}},
     {"Q4_0"}},

    {"handy-computer/multitalker-parakeet-streaming-0.6b-v1-gguf",
     {{"bundle/multitalker-parakeet-streaming-0.6b-v1-F16.gguf"},
      {"bundle/multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf"},
      {"bundle/multitalker-parakeet-streaming-0.6b-v1-Q5_K_M.gguf"},
      {"bundle/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf"}},
     {
      {"", "bundle/multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "bundle/multitalker-parakeet-streaming-0.6b-v1-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "bundle/multitalker-parakeet-streaming-0.6b-v1-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "bundle/multitalker-parakeet-streaming-0.6b-v1-F16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "F16"}},

    {"HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive",
     {{"Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"},
      {"Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"},
      {"Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"},
      {"mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf"}},
     {
      {"", "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf", 1, "mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"IQ3_M", "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf", 1, "mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q8_K_P", "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf", 1, "mmproj-Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_K_P", "Q4_K_P", "IQ3_M"}},

    {"HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
     {{"Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf"},
      {"Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"},
      {"Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"},
      {"mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"}},
     {
      {"", "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"IQ3_M", "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-IQ3_M.gguf", 1, "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q8_K_P", "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf", 1, "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_K_P", "Q4_K_M", "IQ3_M"}},

    {"HauhauCS/GPTOSS-120B-Uncensored-HauhauCS-Aggressive",
     {{"GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf"}},
     {
      {"", "GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf", 1, "", "", "", "", ""},
      {"MXFP4", "GPTOSS-120B-Uncensored-HauhauCS-Aggressive-MXFP4.gguf", 1, "", "", "", "", ""}},
     {"MXFP4"}},

    {"HauhauCS/Qwen3.5-27B-Uncensored-HauhauCS-Aggressive",
     {{"Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"},
      {"Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q3_K_M.gguf"},
      {"Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"},
      {"Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"},
      {"mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf"}},
     {
      {"", "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", 1, "mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"BF16", "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf", 1, "mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"F16", "Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf", 1, "mmproj-Qwen3.5-27B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_M", "BF16"}},

    {"HauhauCS/Qwen3.5-2B-Uncensored-HauhauCS-Aggressive",
     {{"Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-BF16.gguf"},
      {"Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"},
      {"Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf"},
      {"Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"},
      {"mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf"}},
     {
      {"", "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", 1, "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"BF16", "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-BF16.gguf", 1, "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""},
      {"F16", "Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-BF16.gguf", 1, "mmproj-Qwen3.5-2B-Uncensored-HauhauCS-Aggressive-f16.gguf", "", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M", "BF16"}},

    {"HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive",
     {{"Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf"},
      {"Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"},
      {"Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf"},
      {"Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"},
      {"mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf"}},
     {
      {"", "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 1, "mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf", 1, "mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf", "", "", "", ""},
      {"BF16", "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf", 1, "mmproj-Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-BF16.gguf", "", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M", "BF16"}},

    {"Hikari07jp/Ternary-Bonsai-27B-Abliterated-LowDeg-GGUF",
     {{"Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf"}},
     {
      {"", "Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf", 1, "", "", "", "", ""},
      {"Q2_0", "Ternary-Bonsai-27B-Abliterated-LowDeg-Q2_0.gguf", 1, "", "", "", "", ""}},
     {"Q2_0"}},

    {"Hob-forge/Kolibri-1-GGUF",
     {{"Kolibri-1-Q2_K.gguf"},
      {"Kolibri-1-Q4_K_M.gguf"},
      {"Q5_K_M/Kolibri-1-Q5_K_M", 2},
      {"Q8_0/Kolibri-1-Q8_0", 2}},
     {
      {"", "Kolibri-1-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Kolibri-1-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Q8_0/Kolibri-1-Q8_0-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"Q2_K", "Kolibri-1-Q2_K.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "Q2_K"}},

    {"huihui-ai/Huihui-Agents-A1-abliterated-GGUF",
     {{"Agents-A1-abliterated-Q4_K.gguf"},
      {"Agents-A1-abliterated-Q8_0.gguf"},
      {"Agents-A1-abliterated-bf16.gguf"},
      {"mmproj-model-bf16.gguf"}},
     {
      {"", "Agents-A1-abliterated-Q8_0.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"Q8_0", "Agents-A1-abliterated-Q8_0.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"Q4_K", "Agents-A1-abliterated-Q4_K.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"BF16", "Agents-A1-abliterated-bf16.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""}},
     {"BF16", "Q8_0", "Q4_K"}},

    {"huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF",
     {{"UD-IQ1_M/GLM-5.3-Flash-UD-IQ1_M", 3},
      {"UD-Q2_K_XL/GLM-5.3-Flash-UD-Q2_K_XL", 4},
      {"UD-Q6_K_XL/GLM-5.3-Flash-UD-Q6_K_XL", 7},
      {"mmproj-model-bf16.gguf"}},
     {
      {"", "UD-IQ1_M/GLM-5.3-Flash-UD-IQ1_M-00001-of-00003.gguf", 3, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"UD-IQ1_M", "UD-IQ1_M/GLM-5.3-Flash-UD-IQ1_M-00001-of-00003.gguf", 3, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"UD-Q6_K_XL", "UD-Q6_K_XL/GLM-5.3-Flash-UD-Q6_K_XL-00001-of-00007.gguf", 7, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q6_K_XL", "Q2_K_XL", "IQ1_M"}},

    {"huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF",
     {{"BF16/Qwen3.8-Flash-Next-BF16", 8},
      {"Q8_0/Qwen3.8-Flash-Next-Q8_0", 6},
      {"mmproj-Qwen3.8-Flash-Next-BF16.gguf"},
      {"mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"},
      {"mmproj-model-bf16.gguf"},
      {"swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS", 2},
      {"unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL", 4}},
     {
      {"", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf", 6, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf", 6, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf", 8, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"UD-Q4_K_XL", "unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", 4, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""}},
     {"Q4_K_XL", "IQ3_XXS", "Q8_0", "BF16"}},

    {"huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
     {{"GGUF/ggml-model-Q4_K_M.gguf"},
      {"GGUF/ggml-model-f16.gguf"},
      {"GGUF/ggml-model-q8_0.gguf"},
      {"GGUF/mmproj-ggml-model-f16.gguf"}},
     {
      {"", "GGUF/ggml-model-Q4_K_M.gguf", 1, "GGUF/mmproj-ggml-model-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "GGUF/ggml-model-Q4_K_M.gguf", 1, "GGUF/mmproj-ggml-model-f16.gguf", "", "", "", ""},
      {"Q8_0", "GGUF/ggml-model-q8_0.gguf", 1, "GGUF/mmproj-ggml-model-f16.gguf", "", "", "", ""}},
     {"Q8_0", "F16", "Q4_K_M"}},

    {"ibm-granite/granite-4.2-30b-GGUF",
     {{"granite-4.2-30b-Q2_K.gguf"},
      {"granite-4.2-30b-Q4_K_M.gguf"},
      {"granite-4.2-30b-Q4_K_S.gguf"},
      {"granite-4.2-30b-Q8_0.gguf"},
      {"granite-4.2-30b-bf16", 2}},
     {
      {"", "granite-4.2-30b-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "granite-4.2-30b-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "granite-4.2-30b-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "granite-4.2-30b-Q2_K.gguf", 1, "", "", "", "", ""},
      {"BF16", "granite-4.2-30b-bf16-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"BF16", "Q8_0", "Q4_K_S", "Q4_K_M", "Q2_K"}},

    {"Isi99999/LTX-Video",
     {{"LTX_2_3/ltx-2.3-22b-dev-Q4_K_M.gguf"}},
     {
      {"", "LTX_2_3/ltx-2.3-22b-dev-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "LTX_2_3/ltx-2.3-22b-dev-Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
     {{"IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M", 2},
      {"mmproj-Qwen3.8-Flash-Next-BF16.gguf"}},
     {
      {"", "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"IQ1_M", "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"IQ1_M"}},

    {"ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
     {{"IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS", 2},
      {"IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS", 2},
      {"Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0", 2},
      {"mmproj-Qwen3.8-Flash-Next-BF16.gguf"}},
     {
      {"", "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"IQ2_XS", "IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"Q2_0", "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q2_0", "IQ3_XXS", "IQ2_XS"}},

    {"Jackrong/Qwopus3.6-27B-v1-preview-GGUF",
     {{"Qwopus3.6-27B-v1-preview-Q4_K_M.gguf"},
      {"Qwopus3.6-27B-v1-preview-Q6_K.gguf"},
      {"Qwopus3.6-27B-v1-preview-Q8_0.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwopus3.6-27B-v1-preview-Q4_K_M.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwopus3.6-27B-v1-preview-Q4_K_M.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"Q8_0", "Qwopus3.6-27B-v1-preview-Q8_0.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"F32", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M"}},

    {"Jackrong/Qwopus3.8-27B-Flash-V2-GGUF",
     {{"Qwopus3.8-27B-Flash-V2-MTP-BF16.gguf"},
      {"Qwopus3.8-27B-Flash-V2-MTP-Q4_K_M.gguf"},
      {"Qwopus3.8-27B-Flash-V2-MTP-Q8_0.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwopus3.8-27B-Flash-V2-MTP-Q4_K_M.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwopus3.8-27B-Flash-V2-MTP-Q4_K_M.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"Q8_0", "Qwopus3.8-27B-Flash-V2-MTP-Q8_0.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"BF16", "Qwopus3.8-27B-Flash-V2-MTP-BF16.gguf", 1, "mmproj-F32.gguf", "", "", "", ""},
      {"F32", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "BF16"}},

    {"JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q4_K_M",
     {{"Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf"}},
     {
      {"", "Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Mellum2-12B-A2.5B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"JetBrains/Mellum2-12B-A2.5B-Instruct-GGUF-Q8_0",
     {{"Mellum2-12B-A2.5B-Instruct-Q8_0.gguf"}},
     {
      {"", "Mellum2-12B-A2.5B-Instruct-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Mellum2-12B-A2.5B-Instruct-Q8_0.gguf", 1, "", "", "", "", ""}},
     {"Q8_0"}},

    {"JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF",
     {{"Mellum2.1-12B-A2.5B-Thinking-BF16.gguf"},
      {"Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf"},
      {"Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf"}},
     {
      {"", "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Mellum2.1-12B-A2.5B-Thinking-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "Mellum2.1-12B-A2.5B-Thinking-BF16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "BF16"}},

    {"JonathanColetti/Qwen3.8-27B-Uncensored-GGUF",
     {{"Qwen3.8-27B-Uncensored-IQ2_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q5_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q8_0.gguf"},
      {"Qwen3.8-27B-Uncensored-draft-Q4_0.gguf"},
      {"Qwen3.8-27B-Uncensored-draft-Q8_0.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-Q5_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-noMTP-Q8_0.gguf"},
      {"mmproj-Qwen3.8-27B-Uncensored-F16.gguf"}},
     {
      {"", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.8-27B-Uncensored-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"IQ2_M", "Qwen3.8-27B-Uncensored-IQ2_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"Q4_0", "Qwen3.8-27B-Uncensored-draft-Q4_0.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""},
      {"mtp", "Qwen3.8-27B-Uncensored-noMTP-IQ2_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-F16.gguf", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "IQ2_M", "Q4_0"}},

    {"kai-os/Carnice-V2-27b-GGUF",
     {{"carnice-v2-27b-IQ2_M.gguf"},
      {"carnice-v2-27b-Q4_K_M.gguf"},
      {"carnice-v2-27b-Q5_K_M.gguf"},
      {"carnice-v2-27b-Q8_0.gguf"},
      {"carnice-v2-27b-bf16.gguf"}},
     {
      {"", "carnice-v2-27b-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "carnice-v2-27b-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "carnice-v2-27b-Q8_0.gguf", 1, "", "", "", "", ""},
      {"IQ2_M", "carnice-v2-27b-IQ2_M.gguf", 1, "", "", "", "", ""},
      {"BF16", "carnice-v2-27b-bf16.gguf", 1, "", "", "", "", ""}},
     {"BF16", "Q8_0", "Q5_K_M", "Q4_K_M", "IQ2_M"}},

    {"KBLab/sentence-bert-swedish-cased",
     {{"Sentence-Bert-Swedish-Cased-124M-BF16.gguf"}},
     {
      {"", "Sentence-Bert-Swedish-Cased-124M-BF16.gguf", 1, "", "", "", "", ""},
      {"BF16", "Sentence-Bert-Swedish-Cased-124M-BF16.gguf", 1, "", "", "", "", ""}},
     {"BF16"}},

    {"KBlueLeaf/DanTagGen-delta-rev2",
     {{"ggml-model-Q6_K.gguf"},
      {"ggml-model-Q8_0.gguf"},
      {"ggml-model-f16.gguf"}},
     {
      {"", "ggml-model-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "ggml-model-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q6_K", "ggml-model-Q6_K.gguf", 1, "", "", "", "", ""},
      {"F16", "ggml-model-f16.gguf", 1, "", "", "", "", ""}},
     {"F16", "Q8_0", "Q6_K"}},

    {"KittenML/kitten-tts-2",
     {{"cpp/model-tq2_1.gguf"}},
     {
      {"", "cpp/model-tq2_1.gguf", 1, "", "", "", "", ""},
      {"TQ2_1", "cpp/model-tq2_1.gguf", 1, "", "", "", "", ""}},
     {"TQ2_1"}},

    {"LiquidAI/LFM2.5-2.6B-DSpark-GGUF",
     {{"LFM2.5-2.6B-DSpark-F16.gguf"},
      {"LFM2.5-2.6B-DSpark-Q4_K_M.gguf"},
      {"LFM2.5-2.6B-DSpark-Q8_0.gguf"}},
     {
      {"", "LFM2.5-2.6B-DSpark-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "LFM2.5-2.6B-DSpark-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "LFM2.5-2.6B-DSpark-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "LFM2.5-2.6B-DSpark-F16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "F16"}},

    {"LiquidAI/LFM2.5-Audio-1.5B-GGUF",
     {{"LFM2.5-Audio-1.5B-F16.gguf"},
      {"LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"LFM2.5-Audio-1.5B-Q8_0.gguf"},
      {"mmproj-LFM2.5-Audio-1.5B-F16.gguf"},
      {"mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf"},
      {"tokenizer-LFM2.5-Audio-1.5B-F16.gguf"},
      {"tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"tokenizer-LFM2.5-Audio-1.5B-Q8_0.gguf"},
      {"vocoder-LFM2.5-Audio-1.5B-F16.gguf"},
      {"vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf"},
      {"vocoder-LFM2.5-Audio-1.5B-Q8_0.gguf"}},
     {
      {"", "LFM2.5-Audio-1.5B-Q8_0.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "LFM2.5-Audio-1.5B-Q8_0.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-Q8_0.gguf", "", "", "", ""},
      {"F16", "LFM2.5-Audio-1.5B-F16.gguf", 1, "mmproj-LFM2.5-Audio-1.5B-F16.gguf", "", "", "", ""}},
     {"Q8_0", "Q4_0", "F16"}},

    {"llm-jp/llm-jp-4-33b-thinking-gguf",
     {{"llm-jp-4-33b-thinking-BF16.gguf"},
      {"llm-jp-4-33b-thinking-Q4_K_M.gguf"}},
     {
      {"", "llm-jp-4-33b-thinking-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "llm-jp-4-33b-thinking-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"BF16", "llm-jp-4-33b-thinking-BF16.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M", "BF16"}},

    {"lmg-anon/vntl-llama3-8b-v2-gguf",
     {{"vntl-llama3-8b-v2-hf-q5_k_m.gguf"},
      {"vntl-llama3-8b-v2-hf-q8_0.gguf"}},
     {
      {"", "vntl-llama3-8b-v2-hf-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "vntl-llama3-8b-v2-hf-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q5_K_M", "vntl-llama3-8b-v2-hf-q5_k_m.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M"}},

    {"lmstudio-ai/gemma-2b-it-GGUF",
     {{"gemma-2b-it-q4_k_m.gguf"},
      {"gemma-2b-it-q8_0.gguf"}},
     {
      {"", "gemma-2b-it-q4_k_m.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "gemma-2b-it-q4_k_m.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "gemma-2b-it-q8_0.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M"}},

    {"lmstudio-community/Bonsai-27B-GGUF",
     {{"Bonsai-27B-Q1_0.gguf"},
      {"mmproj-Bonsai-27B-BF16.gguf"}},
     {
      {"", "Bonsai-27B-Q1_0.gguf", 1, "mmproj-Bonsai-27B-BF16.gguf", "", "", "", ""},
      {"Q1_0", "Bonsai-27B-Q1_0.gguf", 1, "mmproj-Bonsai-27B-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q1_0"}},

    {"lmstudio-community/GLM-4.6V-Flash-GGUF",
     {{"GLM-4.6V-Flash-Q4_K_M.gguf"},
      {"GLM-4.6V-Flash-Q6_K.gguf"},
      {"GLM-4.6V-Flash-Q8_0.gguf"},
      {"mmproj-GLM-4.6V-Flash-F16.gguf"}},
     {
      {"", "GLM-4.6V-Flash-Q4_K_M.gguf", 1, "mmproj-GLM-4.6V-Flash-F16.gguf", "", "", "", ""},
      {"Q4_K_M", "GLM-4.6V-Flash-Q4_K_M.gguf", 1, "mmproj-GLM-4.6V-Flash-F16.gguf", "", "", "", ""},
      {"Q8_0", "GLM-4.6V-Flash-Q8_0.gguf", 1, "mmproj-GLM-4.6V-Flash-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M"}},

    {"lmstudio-community/gpt-oss-120b-GGUF",
     {{"gpt-oss-120b-MXFP4", 2}},
     {
      {"", "gpt-oss-120b-MXFP4-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"MXFP4", "gpt-oss-120b-MXFP4-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"MXFP4"}},

    {"lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
     {{"Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"},
      {"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"},
      {"Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf"},
      {"Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"}},
     {
      {"", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", 1, "", "", "", "", ""},
      {"IQ4_XS", "Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q5_K_M", "Q4_K_M", "IQ4_XS"}},

    {"lmstudio-community/Muse-Glimmer-30B-GGUF",
     {{"Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"},
      {"mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"}},
     {
      {"", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", 1, "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", "", "", "", ""},
      {"Q4_K_M", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", 1, "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", "", "", "", ""}},
     {"Q4_K_M"}},

    {"lmstudio-community/Qwen3-Coder-Next-GGUF",
     {{"Qwen3-Coder-Next-Q4_K_M.gguf"},
      {"Qwen3-Coder-Next-Q6_K", 2},
      {"Qwen3-Coder-Next-Q8_0", 3}},
     {
      {"", "Qwen3-Coder-Next-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen3-Coder-Next-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Qwen3-Coder-Next-Q8_0-00001-of-00003.gguf", 3, "", "", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M"}},

    {"LocalAI-io/privacy-filter-multilingual-GGUF",
     {{"privacy-filter-multilingual-f16.gguf"},
      {"privacy-filter-multilingual-q8.gguf"}},
     {
      {"", "privacy-filter-multilingual-f16.gguf", 1, "", "", "", "", ""},
      {"F16", "privacy-filter-multilingual-f16.gguf", 1, "", "", "", "", ""},
      {"Q8", "privacy-filter-multilingual-q8.gguf", 1, "", "", "", "", ""}},
     {"Q8", "F16"}},

    {"meta-models/Muse-Glimmer-30B-GGUF",
     {{"Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"},
      {"Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf"},
      {"dflash-Muse-Glimmer-30B-Q4_K_M.gguf"},
      {"mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"}},
     {
      {"", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", 1, "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", "", "dflash-Muse-Glimmer-30B-Q4_K_M.gguf", "", ""},
      {"Q4_K_M", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", 1, "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", "", "dflash-Muse-Glimmer-30B-Q4_K_M.gguf", "", ""},
      {"Q4_K_XL", "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf", 1, "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf", "", "dflash-Muse-Glimmer-30B-Q4_K_M.gguf", "", ""}},
     {"Q4_K_XL", "Q4_K_M"}},

    {"migtissera/Tess-4-27B-GGUF",
     {{"Tess-4-27B-Q4_K_M.gguf"},
      {"Tess-4-27B-Q6_K.gguf"},
      {"Tess-4-27B-Q8_0.gguf"},
      {"mmproj-Tess-4-27B-F16.gguf"},
      {"mtp-Tess-4-27B-Q4_K_M.gguf"},
      {"mtp-Tess-4-27B-Q8_0.gguf"}},
     {
      {"", "Tess-4-27B-Q4_K_M.gguf", 1, "mmproj-Tess-4-27B-F16.gguf", "mtp-Tess-4-27B-Q4_K_M.gguf", "", "", ""},
      {"Q4_K_M", "Tess-4-27B-Q4_K_M.gguf", 1, "mmproj-Tess-4-27B-F16.gguf", "mtp-Tess-4-27B-Q4_K_M.gguf", "", "", ""},
      {"Q8_0", "Tess-4-27B-Q8_0.gguf", 1, "mmproj-Tess-4-27B-F16.gguf", "mtp-Tess-4-27B-Q8_0.gguf", "", "", ""},
      {"F16", "", 0, "", "mtp-Tess-4-27B-Q8_0.gguf", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M"}},

    {"mixedbread-ai/mxbai-embed-large-v1",
     {{"gguf/mxbai-embed-large-v1-f16.gguf"}},
     {
      {"", "gguf/mxbai-embed-large-v1-f16.gguf", 1, "", "", "", "", ""},
      {"F16", "gguf/mxbai-embed-large-v1-f16.gguf", 1, "", "", "", "", ""}},
     {"F16"}},

    {"mondk/MiniCPM5-2B-Abliterated-Uncensored-GGUF",
     {{"minicpm5-2b-abliterated-F16.gguf"},
      {"minicpm5-2b-abliterated-Q4_K_M.gguf"},
      {"minicpm5-2b-abliterated-Q8_0.gguf"}},
     {
      {"", "minicpm5-2b-abliterated-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "minicpm5-2b-abliterated-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "minicpm5-2b-abliterated-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "minicpm5-2b-abliterated-F16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "F16"}},

    {"NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF",
     {{"K2-Horizon-MoVA-36B-A4B-BF16.gguf"},
      {"K2-Horizon-MoVA-36B-A4B-Q2_0.gguf"},
      {"K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf"},
      {"K2-Horizon-MoVA-36B-A4B-Q8_0.gguf"}},
     {
      {"", "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "K2-Horizon-MoVA-36B-A4B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "K2-Horizon-MoVA-36B-A4B-BF16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q2_0", "BF16"}},

    {"ngquocvinh/AliceAI-T5-35B-A0.6B-GGUF",
     {{"AliceAI-T5-35B-A0.6B-BF16.gguf"},
      {"AliceAI-T5-35B-A0.6B-Q3_K_S.gguf"},
      {"AliceAI-T5-35B-A0.6B-Q4_K_M.gguf"},
      {"AliceAI-T5-35B-A0.6B-Q8_0.gguf"},
      {"AliceAI-T5-35B-A0.6B-TQ2_0.gguf"}},
     {
      {"", "AliceAI-T5-35B-A0.6B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "AliceAI-T5-35B-A0.6B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "AliceAI-T5-35B-A0.6B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "AliceAI-T5-35B-A0.6B-BF16.gguf", 1, "", "", "", "", ""},
      {"TQ2_0", "AliceAI-T5-35B-A0.6B-TQ2_0.gguf", 1, "", "", "", "", ""}},
     {"TQ2_0", "Q8_0", "Q4_K_M", "Q3_K_S", "BF16"}},

    {"ngquocvinh/NeoHorse-1-4B-GGUF",
     {{"NeoHorse-1-4B-IQ1_M.gguf"},
      {"NeoHorse-1-4B-Q2_K_S.gguf"},
      {"NeoHorse-1-4B-Q4_K_M.gguf"},
      {"NeoHorse-1-4B-Q8_0.gguf"},
      {"NeoHorse-1-4B-TQ2_0.gguf"}},
     {
      {"", "NeoHorse-1-4B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "NeoHorse-1-4B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "NeoHorse-1-4B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"IQ1_M", "NeoHorse-1-4B-IQ1_M.gguf", 1, "", "", "", "", ""},
      {"TQ2_0", "NeoHorse-1-4B-TQ2_0.gguf", 1, "", "", "", "", ""}},
     {"TQ2_0", "Q8_0", "Q4_K_M", "Q2_K_S", "IQ1_M"}},

    {"nif0/Qwen3-VL-32B-Instruct-MiniMax-H3-GGUF",
     {{"Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-IQ4_XS.gguf"},
      {"Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-Q4_K_M.gguf"},
      {"Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-UD-IQ3_XXS.gguf"},
      {"Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-F32.gguf"}},
     {
      {"", "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-Q4_K_M.gguf", 1, "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-F32.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-Q4_K_M.gguf", 1, "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-F32.gguf", "", "", "", ""},
      {"IQ4_XS", "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-IQ4_XS.gguf", 1, "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-F32.gguf", "", "", "", ""},
      {"F32", "", 0, "", "", "", "", ""}},
     {"IQ3_XXS", "Q4_K_M", "IQ4_XS"}},

    {"NikolayKozloff/Meta-Llama-3.1-8B-Instruct-Q8_0-GGUF",
     {{"meta-llama-3.1-8b-instruct-q8_0.gguf"}},
     {
      {"", "meta-llama-3.1-8b-instruct-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "meta-llama-3.1-8b-instruct-q8_0.gguf", 1, "", "", "", "", ""}},
     {"Q8_0"}},

    {"noctrex/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
     {{"Huihui-Qwen3-VL-4B-Instruct-abliterated-BF16.gguf"},
      {"Huihui-Qwen3-VL-4B-Instruct-abliterated-IQ4_XS.gguf"},
      {"Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf"},
      {"Huihui-Qwen3-VL-4B-Instruct-abliterated-Q8_0.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Huihui-Qwen3-VL-4B-Instruct-abliterated-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "Huihui-Qwen3-VL-4B-Instruct-abliterated-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "Huihui-Qwen3-VL-4B-Instruct-abliterated-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "IQ4_XS", "BF16"}},

    {"OBLITERATUS/Gemma-4-12B-OBLITERATED",
     {{"Gemma-4-12B-OBLITERATED-BF16.gguf"},
      {"Gemma-4-12B-OBLITERATED-Q4_K_M.gguf"},
      {"Gemma-4-12B-OBLITERATED-v2-Q8_0.gguf"}},
     {
      {"", "Gemma-4-12B-OBLITERATED-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Gemma-4-12B-OBLITERATED-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Gemma-4-12B-OBLITERATED-v2-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "Gemma-4-12B-OBLITERATED-BF16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "BF16"}},

    {"OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
     {{"gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf"},
      {"gemma-4-E4B-it-OBLITERATED-Q8_0.gguf"},
      {"gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf"}},
     {
      {"", "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf", 1, "gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf", 1, "gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf", "", "", "", ""},
      {"Q8_0", "gemma-4-E4B-it-OBLITERATED-Q8_0.gguf", 1, "gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M"}},

    {"OBLITERATUS/Qwen3.8-27B-OBLITERATED",
     {{"Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf"},
      {"Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf"},
      {"Qwen3.8-27B-OBLITERATED-Q8_0.gguf"},
      {"mmproj-model-bf16.gguf"}},
     {
      {"", "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.8-27B-OBLITERATED-Q8_0.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"IQ4_XS", "Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf", 1, "mmproj-model-bf16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "IQ4_XS"}},

    {"openthaigpt/openthaigpt1.5-7b-instruct",
     {{"openthaigpt1.5-7B-F16.gguf"},
      {"openthaigpt1.5-7B-instruct-Q3KM.gguf"},
      {"openthaigpt1.5-7B-instruct-Q4KM.gguf"}},
     {
      {"", "openthaigpt1.5-7B-F16.gguf", 1, "", "", "", "", ""},
      {"F16", "openthaigpt1.5-7B-F16.gguf", 1, "", "", "", "", ""},
      {"Q4KM", "openthaigpt1.5-7B-instruct-Q4KM.gguf", 1, "", "", "", "", ""}},
     {"Q4KM", "Q3KM", "F16"}},

    {"orcarouter/Qwen3.8-27B-Uncensored-GGUF",
     {{"Qwen3.8-27B-Uncensored-F16", 2},
      {"Qwen3.8-27B-Uncensored-Q3_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Q8_0.gguf"},
      {"mmproj-Qwen3.8-27B-Uncensored-f16.gguf"}},
     {
      {"", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-f16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.8-27B-Uncensored-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-f16.gguf", "", "", "", ""},
      {"F16", "Qwen3.8-27B-Uncensored-F16-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-27B-Uncensored-f16.gguf", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_M", "F16"}},

    {"oruk/orukeet",
     {{"orukeet-transcribe-cpp-Q8_0.gguf"},
      {"orukeet-v0.1.0-q8.gguf"},
      {"transcribe-cpp/orukeet-Q8_0.gguf"}},
     {
      {"", "orukeet-transcribe-cpp-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "orukeet-transcribe-cpp-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8", "orukeet-v0.1.0-q8.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q8"}},

    {"peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP",
     {{"Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf"},
      {"Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf"},
      {"Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q8_K_XL.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-Q8_0.gguf"}},
     {
      {"", "Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf", 1, "mmproj-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "", 0, "", "", "", "", ""},
      {"UD-IQ3_XXS", "Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf", 1, "mmproj-Q8_0.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q8_K_XL.gguf", 1, "mmproj-Q8_0.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_K_XL", "Q4_K_S", "IQ3_XXS"}},

    {"peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF",
     {{"Dirk-Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf"},
      {"Dirk-Qwen3.8-27B-UD-Q4_K_S.gguf"},
      {"Dirk-Qwen3.8-27B-UD-Q8_K_XL.gguf"},
      {"mmproj-F16.gguf"}},
     {
      {"", "Dirk-Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"IQ2_S", "Dirk-Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "Dirk-Qwen3.8-27B-UD-Q8_K_XL.gguf", 1, "mmproj-F16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q8_K_XL", "Q4_K_S", "IQ2_S"}},

    {"peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP",
     {{"Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf"},
      {"Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf"},
      {"Tiel-Coder-35B-A3B-MTP-UD-Q8_K_XL.gguf"},
      {"mmproj-BF16.gguf"}},
     {
      {"", "Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-IQ3_XXS", "Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "Tiel-Coder-35B-A3B-MTP-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_K_XL", "Q4_K_S", "IQ3_XXS"}},

    {"perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-gguf-20260826",
     {{"draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"},
      {"draft/qwen38-27b-mtp-Q8_0.gguf"},
      {"qwen38-27b-Q4_K_M.gguf"},
      {"qwen38-27b-mmproj-f16.gguf"}},
     {
      {"", "draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", 1, "qwen38-27b-mmproj-f16.gguf", "draft/qwen38-27b-mtp-Q8_0.gguf", "", "", ""},
      {"Q4_K_M", "draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", 1, "qwen38-27b-mmproj-f16.gguf", "draft/qwen38-27b-mtp-Q8_0.gguf", "", "", ""},
      {"Q8_0", "", 0, "", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"philbert440/Qwen3.8-27B-Uncensored-Cyber-GGUF",
     {{"Qwen3.8-27B-Uncensored-Cyber-Q4_K_M.gguf"},
      {"Qwen3.8-27B-Uncensored-Cyber-Q6_K.gguf"},
      {"Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf"},
      {"mmproj-Qwen3.8-27B-Uncensored-Cyber-BF16.gguf"},
      {"mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf"},
      {"mtp-Qwen3.8-27B-Uncensored-Cyber-BF16.gguf"},
      {"mtp-Qwen3.8-27B-Uncensored-Cyber-Q4_0.gguf"},
      {"mtp-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf"}},
     {
      {"", "Qwen3.8-27B-Uncensored-Cyber-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", "mtp-Qwen3.8-27B-Uncensored-Cyber-Q4_0.gguf", "", "", ""},
      {"Q4_K_M", "Qwen3.8-27B-Uncensored-Cyber-Q4_K_M.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", "mtp-Qwen3.8-27B-Uncensored-Cyber-Q4_0.gguf", "", "", ""},
      {"Q8_0", "Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", 1, "mmproj-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", "mtp-Qwen3.8-27B-Uncensored-Cyber-Q8_0.gguf", "", "", ""},
      {"BF16", "", 0, "", "mtp-Qwen3.8-27B-Uncensored-Cyber-BF16.gguf", "", "", ""}},
     {"Q8_0", "Q6_K", "Q4_K_M"}},

    {"ProCreations/grug-27b-qat-q4-gguf",
     {{"grug-27b-qat-Q4_K_M.gguf"},
      {"mmproj-grug-27b-f16.gguf"}},
     {
      {"", "grug-27b-qat-Q4_K_M.gguf", 1, "mmproj-grug-27b-f16.gguf", "", "", "", ""},
      {"Q4_K_M", "grug-27b-qat-Q4_K_M.gguf", 1, "mmproj-grug-27b-f16.gguf", "", "", "", ""},
      {"F16", "", 0, "", "", "", "", ""}},
     {"Q4_K_M"}},

    {"QuantaPlanta/MiMo-V2.6-Flash-MOPD-MXFP4-GGUF",
     {{"MiMo-V2.6-Flash-MOPD-MXFP4.gguf"},
      {"mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf"},
      {"mmproj-MiMo-V2.6-Flash-MOPD-bf16.gguf"},
      {"mmproj-MiMo-V2.6-Flash-MOPD-f16.gguf"}},
     {
      {"", "MiMo-V2.6-Flash-MOPD-MXFP4.gguf", 1, "mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "", 0, "", "", "", "", ""},
      {"MXFP4", "MiMo-V2.6-Flash-MOPD-MXFP4.gguf", 1, "mmproj-MiMo-V2.6-Flash-MOPD-Q8_0.gguf", "", "", "", ""}},
     {"MXFP4"}},

    {"Qwen/Qwen3-VL-2B-Instruct-GGUF",
     {{"Qwen3VL-2B-Instruct-F16.gguf"},
      {"Qwen3VL-2B-Instruct-Q4_K_M.gguf"},
      {"Qwen3VL-2B-Instruct-Q8_0.gguf"},
      {"mmproj-Qwen3VL-2B-Instruct-F16.gguf"},
      {"mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"}},
     {
      {"", "Qwen3VL-2B-Instruct-Q4_K_M.gguf", 1, "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3VL-2B-Instruct-Q4_K_M.gguf", 1, "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3VL-2B-Instruct-Q8_0.gguf", 1, "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf", "", "", "", ""},
      {"F16", "Qwen3VL-2B-Instruct-F16.gguf", 1, "mmproj-Qwen3VL-2B-Instruct-F16.gguf", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "F16"}},

    {"ReadyArt/Serenity-26B-A4B-GGUF",
     {{"Serenity-26B-A4B-HB16-Q6_K.gguf"},
      {"Serenity-26B-A4B-HB16-Q8_0.gguf"},
      {"Serenity-26B-A4B-IQ4_XS.gguf"},
      {"Serenity-26B-A4B-Q4_K_M.gguf"},
      {"Serenity-26B-A4B-Q5_K_M.gguf"},
      {"Serenity-26B-A4B-Q6_K.gguf"},
      {"Serenity-26B-A4B-Q8_0.gguf"}},
     {
      {"", "Serenity-26B-A4B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Serenity-26B-A4B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Serenity-26B-A4B-HB16-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q6_K", "Serenity-26B-A4B-HB16-Q6_K.gguf", 1, "", "", "", "", ""},
      {"Q5_K_M", "Serenity-26B-A4B-Q5_K_M.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS"}},

    {"realrebelai/KREA-2_GGUFs",
     {{"BASE/Krea-2-Base-Q3_K_M.gguf"},
      {"BASE/Krea-2-Base-Q4_K_M.gguf"},
      {"BASE/Krea-2-Base-Q8_0.gguf"},
      {"TURBO/Krea-2-Turbo-Q4_K_M.gguf"},
      {"TURBO/Krea-2-Turbo-Q8_0.gguf"}},
     {
      {"", "BASE/Krea-2-Base-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "BASE/Krea-2-Base-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "BASE/Krea-2-Base-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q3_K_M", "BASE/Krea-2-Base-Q3_K_M.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_M"}},

    {"realrebelai/MiniMax-H3_GGUFs",
     {{"MiniMax-H3-FL2VA-Q3_K_M.gguf"},
      {"MiniMax-H3-FL2VA-Q4_K_M.gguf"},
      {"MiniMax-H3-REF2VA-Q3_K_M.gguf"},
      {"MiniMax-H3-REF2VA-Q4_K_M.gguf"},
      {"qwen3vl-32B-MiniMax-H3-Q2_K.gguf"},
      {"qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"}},
     {
      {"", "MiniMax-H3-FL2VA-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "MiniMax-H3-FL2VA-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q3_K_M", "MiniMax-H3-FL2VA-Q3_K_M.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "qwen3vl-32B-MiniMax-H3-Q2_K.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M", "Q2_K", "Q3_K_M"}},

    {"realrebelai/Wan-Animate-2_GGUFs",
     {{"Distilled/Wan-Animate-2-14B-TURBO-Q2_K.gguf"},
      {"Distilled/Wan-Animate-2-14B-TURBO-Q4_K_M.gguf"},
      {"Distilled/Wan-Animate-2-14B-TURBO-Q4_K_S.gguf"},
      {"Distilled/Wan-Animate-2-14B-TURBO-Q8_0.gguf"}},
     {
      {"", "Distilled/Wan-Animate-2-14B-TURBO-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Distilled/Wan-Animate-2-14B-TURBO-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Distilled/Wan-Animate-2-14B-TURBO-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q2_K", "Distilled/Wan-Animate-2-14B-TURBO-Q2_K.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_S", "Q4_K_M", "Q2_K"}},

    {"SC117/gemma-4-26B-A4B-it-qat-heretic-GGUF",
     {{"gemma-4-26B-A4B-it-qat-heretic-UD-Q4_K_XL.gguf"},
      {"gemma-4-26B-A4B-it-qat-heretic-f16.gguf"},
      {"mmproj-gemma-4-26B-A4B-it-qat-heretic-F16.gguf"}},
     {
      {"", "gemma-4-26B-A4B-it-qat-heretic-UD-Q4_K_XL.gguf", 1, "mmproj-gemma-4-26B-A4B-it-qat-heretic-F16.gguf", "", "", "", ""},
      {"UD-Q4_K_XL", "gemma-4-26B-A4B-it-qat-heretic-UD-Q4_K_XL.gguf", 1, "mmproj-gemma-4-26B-A4B-it-qat-heretic-F16.gguf", "", "", "", ""},
      {"F16", "gemma-4-26B-A4B-it-qat-heretic-f16.gguf", 1, "mmproj-gemma-4-26B-A4B-it-qat-heretic-F16.gguf", "", "", "", ""}},
     {"F16", "Q4_K_XL"}},

    {"SC117/Spark-X2.5-4B-abliterated-FIT-GGUF",
     {{"Spark-X2.5-4B-abliterated-BF16.gguf"},
      {"Spark-X2.5-4B-abliterated-FIT-COMPACT-2.43GiB-Q4_K.gguf"},
      {"Spark-X2.5-4B-abliterated-FIT-QUALITY-3.15GiB-Q6_K.gguf"}},
     {
      {"", "Spark-X2.5-4B-abliterated-BF16.gguf", 1, "", "", "", "", ""},
      {"BF16", "Spark-X2.5-4B-abliterated-BF16.gguf", 1, "", "", "", "", ""},
      {"Q6_K", "Spark-X2.5-4B-abliterated-FIT-QUALITY-3.15GiB-Q6_K.gguf", 1, "", "", "", "", ""}},
     {"Q6_K", "Q4_K", "BF16"}},

    {"SC117/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF",
     {{"IQ2_XS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS", 2},
      {"Q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-Q2_0", 2},
      {"Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S", 2},
      {"mmproj-Qwen3.8-Flash-Next-BF16.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q8_0.gguf"},
      {"mtp-q2_0.gguf"}},
     {
      {"", "IQ2_XS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mtp-q2_0.gguf", "", "", ""},
      {"Q8_0", "", 0, "", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"IQ2_XS", "IQ2_XS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ2_XS-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mtp-q2_0.gguf", "", "", ""},
      {"IQ3_S", "Swift-Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf", 2, "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "mtp-q2_0.gguf", "", "", ""},
      {"BF16", "", 0, "", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""}},
     {"IQ3_S", "Q2_0", "IQ2_XS"}},

    {"scragnog/MOSS-Music-8B-Instruct-GGUF",
     {{"moss-aud-f16.gguf"},
      {"moss-lm-f16.gguf"},
      {"moss-lm-q8_0.gguf"}},
     {
      {"", "moss-lm-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "moss-lm-q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "moss-aud-f16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "F16"}},

    {"Snowflake/snowflake-arctic-embed-m-v1.5",
     {{"gguf/snowflake-arctic-embed-m-v1.5-bf16.gguf"},
      {"gguf/snowflake-arctic-embed-m-v1.5-q8_0.gguf"},
      {"gguf/snowflake-arctic-embed-m-v1.5-tq2_0.gguf"}},
     {
      {"", "gguf/snowflake-arctic-embed-m-v1.5-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "gguf/snowflake-arctic-embed-m-v1.5-q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "gguf/snowflake-arctic-embed-m-v1.5-bf16.gguf", 1, "", "", "", "", ""},
      {"TQ2_0", "gguf/snowflake-arctic-embed-m-v1.5-tq2_0.gguf", 1, "", "", "", "", ""}},
     {"TQ2_0", "Q8_0", "BF16"}},

    {"Solstice-AI/GLM-5.3-Flash-UNCENSORED-mlx-oQ4e-DFlash2",
     {{"mmproj-BF16.gguf"},
      {"speculative/GLM-5.3-Flash-DFlash2-bf16.gguf"}},
     {
      {"", "speculative/GLM-5.3-Flash-DFlash2-bf16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "speculative/GLM-5.3-Flash-DFlash2-bf16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"BF16"}},

    {"superwhisper/s1-mini-GGUF",
     {{"s1-mini-f16.gguf"},
      {"s1-mini-q4_k_m.gguf"}},
     {
      {"", "s1-mini-q4_k_m.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "s1-mini-q4_k_m.gguf", 1, "", "", "", "", ""},
      {"F16", "s1-mini-f16.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M", "F16"}},

    {"tencent/Hy-MT2-7B-GGUF",
     {{"HY-MT2-7B-Q6_K.gguf"},
      {"HY-MT2-7B-Q8_0.gguf"},
      {"Hy-MT2-7B-Q4_K_M.gguf"}},
     {
      {"", "Hy-MT2-7B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Hy-MT2-7B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "HY-MT2-7B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q6_K", "HY-MT2-7B-Q6_K.gguf", 1, "", "", "", "", ""}},
     {"Q4_K_M", "Q8_0", "Q6_K"}},

    {"TrevorJS/gemma-4-26B-A4B-it-uncensored-GGUF",
     {{"gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf"},
      {"gemma-4-26B-A4B-it-uncensored-Q8_0.gguf"}},
     {
      {"", "gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "gemma-4-26B-A4B-it-uncensored-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "gemma-4-26B-A4B-it-uncensored-Q8_0.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M"}},

    {"ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF",
     {{"IQ1_M/Swift-1.5-Qwen3.8-Flash-Next-IQ1_M", 2},
      {"Q3_K_L/Swift-1.5-Qwen3.8-Flash-Next-Q3_K_L", 3},
      {"Q4_K_M/Swift-1.5-Qwen3.8-Flash-Next-Q4_K_M", 3},
      {"Q8_0/Swift-1.5-Qwen3.8-Flash-Next-Q8_0", 5},
      {"mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf"}},
     {
      {"", "Q4_K_M/Swift-1.5-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf", 3, "mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Q4_K_M/Swift-1.5-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf", 3, "mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/Swift-1.5-Qwen3.8-Flash-Next-Q8_0-00001-of-00005.gguf", 5, "mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"IQ1_M", "IQ1_M/Swift-1.5-Qwen3.8-Flash-Next-IQ1_M-00001-of-00002.gguf", 2, "mmproj-Swift-1.5-Qwen3.8-Flash-Next-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_L", "IQ1_M"}},

    {"unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
     {{"DeepSeek-R1-Distill-Qwen-32B-F16/DeepSeek-R1-Distill-Qwen-32B-F16", 2},
      {"DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"},
      {"DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf"}},
     {
      {"", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "DeepSeek-R1-Distill-Qwen-32B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"F16", "DeepSeek-R1-Distill-Qwen-32B-F16/DeepSeek-R1-Distill-Qwen-32B-F16-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "F16"}},

    {"unsloth/DeepSeek-V4-Flash-0731-GGUF",
     {{"UD-IQ1_M/DeepSeek-V4-Flash-0731-UD-IQ1_M", 3},
      {"UD-IQ4_NL/DeepSeek-V4-Flash-0731-UD-IQ4_NL", 4},
      {"UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL", 5},
      {"dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
      {"dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf"}},
     {
      {"", "UD-IQ1_M/DeepSeek-V4-Flash-0731-UD-IQ1_M-00001-of-00003.gguf", 3, "", "", "", "", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
      {"Q8_0", "", 0, "", "", "", "", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
      {"UD-IQ1_M", "UD-IQ1_M/DeepSeek-V4-Flash-0731-UD-IQ1_M-00001-of-00003.gguf", 3, "", "", "", "", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf", 5, "", "", "", "", "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"}},
     {"Q8_K_XL", "IQ4_NL", "IQ1_M"}},

    {"unsloth/DeepSeek-V4-Flash-GGUF",
     {{"UD-IQ1_M/DeepSeek-V4-Flash-UD-IQ1_M", 3},
      {"UD-IQ4_NL/DeepSeek-V4-Flash-UD-IQ4_NL", 4},
      {"UD-Q8_K_XL/DeepSeek-V4-Flash-UD-Q8_K_XL", 5}},
     {
      {"", "UD-IQ1_M/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf", 3, "", "", "", "", ""},
      {"UD-IQ1_M", "UD-IQ1_M/DeepSeek-V4-Flash-UD-IQ1_M-00001-of-00003.gguf", 3, "", "", "", "", ""},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/DeepSeek-V4-Flash-UD-Q8_K_XL-00001-of-00005.gguf", 5, "", "", "", "", ""}},
     {"Q8_K_XL", "IQ4_NL", "IQ1_M"}},

    {"unsloth/DeepSeek-V4-Pro-0813-GGUF",
     {{"UD-Q4_K_XL/DeepSeek-V4-Pro-0813-UD-Q4_K_XL", 20},
      {"UD-Q8_K_XL/DeepSeek-V4-Pro-0813-UD-Q8_K_XL", 20}},
     {
      {"", "UD-Q4_K_XL/DeepSeek-V4-Pro-0813-UD-Q4_K_XL-00001-of-00020.gguf", 20, "", "", "", "", ""},
      {"UD-Q4_K_XL", "UD-Q4_K_XL/DeepSeek-V4-Pro-0813-UD-Q4_K_XL-00001-of-00020.gguf", 20, "", "", "", "", ""},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/DeepSeek-V4-Pro-0813-UD-Q8_K_XL-00001-of-00020.gguf", 20, "", "", "", "", ""}},
     {"Q8_K_XL", "Q4_K_XL"}},

    {"unsloth/embeddinggemma-2-GGUF",
     {{"embeddinggemma-2-BF16.gguf"},
      {"embeddinggemma-2-Q8_0.gguf"},
      {"embeddinggemma-2-UD-Q4_K_XL.gguf"},
      {"embeddinggemma-2-UD-Q6_K_XL.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-Q8_0.gguf"}},
     {
      {"", "embeddinggemma-2-Q8_0.gguf", 1, "mmproj-Q8_0.gguf", "", "", "", ""},
      {"Q8_0", "embeddinggemma-2-Q8_0.gguf", 1, "mmproj-Q8_0.gguf", "", "", "", ""},
      {"BF16", "embeddinggemma-2-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q6_K_XL", "embeddinggemma-2-UD-Q6_K_XL.gguf", 1, "mmproj-Q8_0.gguf", "", "", "", ""},
      {"F16", "embeddinggemma-2-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q6_K_XL", "Q4_K_XL", "Q8_0", "BF16"}},

    {"unsloth/GLM-4.7-Flash-GGUF",
     {{"BF16/GLM-4.7-Flash-BF16", 2},
      {"GLM-4.7-Flash-Q4_K_M.gguf"},
      {"GLM-4.7-Flash-Q6_K.gguf"},
      {"GLM-4.7-Flash-Q8_0.gguf"},
      {"GLM-4.7-Flash-UD-TQ1_0.gguf"}},
     {
      {"", "GLM-4.7-Flash-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "GLM-4.7-Flash-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "GLM-4.7-Flash-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "BF16/GLM-4.7-Flash-BF16-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"UD-TQ1_0", "GLM-4.7-Flash-UD-TQ1_0.gguf", 1, "", "", "", "", ""}},
     {"TQ1_0", "Q8_0", "Q6_K", "Q4_K_M", "BF16"}},

    {"unsloth/GLM-5.3-Flash-GGUF",
     {{"BF16/GLM-5.3-Flash-BF16", 14},
      {"Q8_0/GLM-5.3-Flash-Q8_0", 8},
      {"UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS", 5},
      {"UD-Q6_K_XL/GLM-5.3-Flash-UD-Q6_K_XL", 7},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"}},
     {
      {"", "Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/GLM-5.3-Flash-BF16-00001-of-00014.gguf", 14, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q6_K_XL", "UD-Q6_K_XL/GLM-5.3-Flash-UD-Q6_K_XL-00001-of-00007.gguf", 7, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/GLM-5.3-Flash-BF16-00001-of-00014.gguf", 14, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q6_K_XL", "IQ4_XS", "Q8_0", "BF16"}},

    {"unsloth/GLM-5.3-GGUF",
     {{"BF16/GLM-5.3-BF16", 33},
      {"Q8_0/GLM-5.3-Q8_0", 17},
      {"UD-IQ4_XS/GLM-5.3-UD-IQ4_XS", 9},
      {"UD-Q6_K_XL/GLM-5.3-UD-Q6_K_XL", 16}},
     {
      {"", "Q8_0/GLM-5.3-Q8_0-00001-of-00017.gguf", 17, "", "", "", "", ""},
      {"Q8_0", "Q8_0/GLM-5.3-Q8_0-00001-of-00017.gguf", 17, "", "", "", "", ""},
      {"BF16", "BF16/GLM-5.3-BF16-00001-of-00033.gguf", 33, "", "", "", "", ""},
      {"UD-Q6_K_XL", "UD-Q6_K_XL/GLM-5.3-UD-Q6_K_XL-00001-of-00016.gguf", 16, "", "", "", "", ""}},
     {"Q6_K_XL", "IQ4_XS", "Q8_0", "BF16"}},

    {"unsloth/inkling-GGUF",
     {{"BF16/inkling-GGUF-BF16", 41},
      {"Q8_0/inkling-GGUF-Q8_0", 19},
      {"UD-IQ1_S/inkling-UD-IQ1_S", 7},
      {"UD-Q4_K_XL/inkling-UD-Q4_K_XL", 14},
      {"mmproj-BF16.gguf"}},
     {
      {"", "Q8_0/inkling-GGUF-Q8_0-00001-of-00019.gguf", 19, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/inkling-GGUF-Q8_0-00001-of-00019.gguf", 19, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/inkling-GGUF-BF16-00001-of-00041.gguf", 41, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q4_K_XL", "UD-Q4_K_XL/inkling-UD-Q4_K_XL-00001-of-00014.gguf", 14, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q4_K_XL", "IQ1_S", "Q8_0", "BF16"}},

    {"unsloth/Kimi-K2.6-GGUF",
     {{"BF16/Kimi-K2.6-BF16", 46},
      {"UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL", 14},
      {"UD-Q8_K_XL/Kimi-K2.6-UD-Q8_K_XL", 14},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf", 46, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf", 46, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/Kimi-K2.6-UD-Q8_K_XL-00001-of-00014.gguf", 14, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Kimi-K2.6-BF16-00001-of-00046.gguf", 46, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "Q4_K_XL", "BF16"}},

    {"unsloth/Kimi-K2.7-Code-GGUF",
     {{"UD-IQ1_M/Kimi-K2.7-Code-UD-IQ1_M", 8},
      {"UD-Q2_K_XL/Kimi-K2.7-Code-UD-Q2_K_XL", 8},
      {"UD-Q8_K_XL/Kimi-K2.7-Code-UD-Q8_K_XL", 14},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "UD-IQ1_M/Kimi-K2.7-Code-UD-IQ1_M-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-IQ1_M", "UD-IQ1_M/Kimi-K2.7-Code-UD-IQ1_M-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/Kimi-K2.7-Code-UD-Q8_K_XL-00001-of-00014.gguf", 14, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q8_K_XL", "Q2_K_XL", "IQ1_M"}},

    {"unsloth/Kimi-K3-GGUF",
     {{"UD-IQ1_M/Kimi-K3-UD-IQ1_M", 15},
      {"UD-Q2_K_XL/Kimi-K3-UD-Q2_K_XL", 19},
      {"UD-TQ2_0/Kimi-K3-UD-TQ2_0", 13},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf", 15, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-IQ1_M", "UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00015.gguf", 15, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-TQ2_0", "UD-TQ2_0/Kimi-K3-UD-TQ2_0-00001-of-00013.gguf", 13, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"TQ2_0", "Q2_K_XL", "IQ1_M"}},

    {"unsloth/LFM2.5-VL-3B-GGUF",
     {{"LFM2.5-VL-3B-BF16.gguf"},
      {"LFM2.5-VL-3B-Q4_K_M.gguf"},
      {"LFM2.5-VL-3B-Q5_K_S.gguf"},
      {"LFM2.5-VL-3B-Q8_0.gguf"},
      {"LFM2.5-VL-3B-UD-Q8_K_XL.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "LFM2.5-VL-3B-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "LFM2.5-VL-3B-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "LFM2.5-VL-3B-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "LFM2.5-VL-3B-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "LFM2.5-VL-3B-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "LFM2.5-VL-3B-BF16.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "Q8_0", "Q5_K_S", "Q4_K_M", "BF16"}},

    {"unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
     {{"BF16/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", 2},
      {"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf"},
      {"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q3_K_XL.gguf"},
      {"NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q8_K_XL.gguf"}},
     {
      {"", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "BF16/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"UD-Q8_K_XL", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q8_K_XL.gguf", 1, "", "", "", "", ""}},
     {"Q8_K_XL", "Q3_K_XL", "Q8_0", "BF16"}},

    {"unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
     {{"BF16/Qwen3-30B-A3B-Instruct-2507-BF16", 2},
      {"Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"},
      {"Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"},
      {"Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf"},
      {"Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf"}},
     {
      {"", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "BF16/Qwen3-30B-A3B-Instruct-2507-BF16-00001-of-00002.gguf", 2, "", "", "", "", ""},
      {"UD-TQ1_0", "Qwen3-30B-A3B-Instruct-2507-UD-TQ1_0.gguf", 1, "", "", "", "", ""}},
     {"TQ1_0", "Q8_0", "Q6_K", "Q4_K_M", "BF16"}},

    {"unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF",
     {{"BF16/Qwen3-VL-30B-A3B-Instruct-BF16", 2},
      {"Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"},
      {"Qwen3-VL-30B-A3B-Instruct-Q6_K.gguf"},
      {"Qwen3-VL-30B-A3B-Instruct-Q8_0.gguf"},
      {"Qwen3-VL-30B-A3B-Instruct-UD-TQ1_0.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3-VL-30B-A3B-Instruct-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Qwen3-VL-30B-A3B-Instruct-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-TQ1_0", "Qwen3-VL-30B-A3B-Instruct-UD-TQ1_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Qwen3-VL-30B-A3B-Instruct-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""}},
     {"TQ1_0", "Q8_0", "Q6_K", "Q4_K_M", "BF16"}},

    {"unsloth/Qwen3.5-27B-GGUF",
     {{"BF16/Qwen3.5-27B-BF16", 2},
      {"Qwen3.5-27B-Q4_K_M.gguf"},
      {"Qwen3.5-27B-Q6_K.gguf"},
      {"Qwen3.5-27B-Q8_0.gguf"},
      {"Qwen3.5-27B-UD-Q8_K_XL.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwen3.5-27B-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.5-27B-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.5-27B-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "Qwen3.5-27B-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Qwen3.5-27B-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "Q8_0", "Q6_K", "Q4_K_M", "BF16"}},

    {"unsloth/Qwen3.5-35B-A3B-GGUF",
     {{"BF16/Qwen3.5-35B-A3B-BF16", 2},
      {"Qwen3.5-35B-A3B-Q4_K_M.gguf"},
      {"Qwen3.5-35B-A3B-Q8_0.gguf"},
      {"Qwen3.5-35B-A3B-UD-IQ3_S.gguf"},
      {"Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwen3.5-35B-A3B-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Qwen3.5-35B-A3B-Q4_K_M.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.5-35B-A3B-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "Qwen3.5-35B-A3B-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "IQ3_S", "Q8_0", "Q4_K_M", "BF16"}},

    {"unsloth/Qwen3.5-397B-A17B-GGUF",
     {{"BF16/Qwen3.5-397B-A17B-BF16", 17},
      {"Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M", 6},
      {"Q8_0/Qwen3.5-397B-A17B-Q8_0", 10},
      {"UD-IQ3_XXS/Qwen3.5-397B-A17B-UD-IQ3_XXS", 4},
      {"UD-Q8_K_XL/Qwen3.5-397B-A17B-UD-Q8_K_XL", 10},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4_K_M", "Q4_K_M/Qwen3.5-397B-A17B-Q4_K_M-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/Qwen3.5-397B-A17B-Q8_0-00001-of-00010.gguf", 10, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Qwen3.5-397B-A17B-BF16-00001-of-00017.gguf", 17, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/Qwen3.5-397B-A17B-UD-Q8_K_XL-00001-of-00010.gguf", 10, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Qwen3.5-397B-A17B-BF16-00001-of-00017.gguf", 17, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "IQ3_XXS", "Q8_0", "Q4_K_M", "BF16"}},

    {"unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
     {{"BF16/Qwen3.6-35B-A3B-BF16", 2},
      {"Qwen3.6-35B-A3B-Q8_0.gguf"},
      {"Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"},
      {"Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf"},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Qwen3.6-35B-A3B-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Qwen3.6-35B-A3B-Q8_0.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf", 2, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "Q3_K_M", "Q8_0", "BF16"}},

    {"unsloth/Qwen3.8-Flash-Next-GGUF",
     {{"BF16/Qwen3.8-Flash-Next-BF16", 8},
      {"MTP/mtp-Qwen3.8-Flash-Next-BF16.gguf"},
      {"MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf"},
      {"MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"},
      {"MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf"},
      {"MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf"},
      {"MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"},
      {"Q8_0/Qwen3.8-Flash-Next-Q8_0", 6},
      {"UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS", 3},
      {"UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL", 6},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mtp-Qwen3.8-Flash-Next-Q8_0.gguf"}},
     {
      {"", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"Q4_K_M", "", 0, "", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"Q8_0", "Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"BF16", "BF16/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""},
      {"UD-Q6_K_XL", "UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf", "", "", ""}},
     {"Q6_K_XL", "IQ4_XS", "Q8_0", "BF16"}},

    {"unsloth/Step-3.7-Flash-GGUF",
     {{"BF16/Step-3.7-Flash-BF16", 8},
      {"Q8_0/Step-3.7-Flash-Q8_0", 6},
      {"UD-Q3_K_M/Step-3.7-Flash-UD-Q3_K_M", 3},
      {"UD-Q8_K_XL/Step-3.7-Flash-UD-Q8_K_XL", 6},
      {"mmproj-BF16.gguf"},
      {"mmproj-F16.gguf"},
      {"mmproj-F32.gguf"}},
     {
      {"", "Q8_0/Step-3.7-Flash-Q8_0-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q8_0", "Q8_0/Step-3.7-Flash-Q8_0-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "BF16/Step-3.7-Flash-BF16-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "", "", "", ""},
      {"UD-Q8_K_XL", "UD-Q8_K_XL/Step-3.7-Flash-UD-Q8_K_XL-00001-of-00006.gguf", 6, "mmproj-BF16.gguf", "", "", "", ""},
      {"F16", "BF16/Step-3.7-Flash-BF16-00001-of-00008.gguf", 8, "mmproj-BF16.gguf", "", "", "", ""}},
     {"Q8_K_XL", "Q3_K_M", "Q8_0", "BF16"}},

    {"vmlinux/Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix-GGUF",
     {{"Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix.gguf"},
      {"mmproj-BF16.gguf"}},
     {
      {"", "Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"Q4MIX", "Qwen3.8-Flash-Next-Uncensored-Gufo-Q4Mix.gguf", 1, "mmproj-BF16.gguf", "", "", "", ""},
      {"BF16", "", 0, "", "", "", "", ""}},
     {"Q4MIX"}},

    {"WalkingCat/Soprano-1.1-80M-GGUF",
     {{"Soprano-1.1-80M-GGUF/soprano-1.1-80m-bf16.gguf"},
      {"Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf"}},
     {
      {"", "Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "Soprano-1.1-80M-GGUF/soprano-1.1-80m-q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "Soprano-1.1-80M-GGUF/soprano-1.1-80m-bf16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "BF16"}},

    {"yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF",
     {{"MTP/gemma-4-12B-it-MTP-BF16.gguf"},
      {"MTP/gemma-4-12B-it-MTP-Q8_0.gguf"},
      {"gemma4-v2-Q3_K_M.gguf"},
      {"gemma4-v2-Q4_K_M.gguf"},
      {"gemma4-v2-Q8_0.gguf"}},
     {
      {"", "gemma4-v2-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "gemma4-v2-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q8_0", "MTP/gemma-4-12B-it-MTP-Q8_0.gguf", 1, "", "", "", "", ""},
      {"BF16", "MTP/gemma-4-12B-it-MTP-BF16.gguf", 1, "", "", "", "", ""}},
     {"Q8_0", "Q4_K_M", "Q3_K_M", "BF16"}},

    {"zetasepic/Qwen2.5-72B-Instruct-abliterated-GGUF",
     {{"Qwen2.5-72B-Instruct-abliterated-Q3_K_M.gguf"},
      {"Qwen2.5-72B-Instruct-abliterated-Q4_K_M.gguf"},
      {"Qwen2.5-72B-Instruct-abliterated-Q6_K", 2}},
     {
      {"", "Qwen2.5-72B-Instruct-abliterated-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q4_K_M", "Qwen2.5-72B-Instruct-abliterated-Q4_K_M.gguf", 1, "", "", "", "", ""},
      {"Q3_K_M", "Qwen2.5-72B-Instruct-abliterated-Q3_K_M.gguf", 1, "", "", "", "", ""},
      {"Q6_K", "Qwen2.5-72B-Instruct-abliterated-Q6_K-00001-of-00002.gguf", 2, "", "", "", "", ""}},
     {"Q6_K", "Q4_K_M", "Q3_K_M"}},
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

static void expect_eq(const std::string & what, const std::string & actual, const std::string & expected) {
    if (actual != expected) {
        fprintf(stderr, "  [%s] %s\n    actual:   '%s'\n    expected: '%s'\n",
                g_context.c_str(), what.c_str(), actual.c_str(), expected.c_str());
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
    expect_eq("parts",   std::to_string(plan.model_files.size()), std::to_string(p.parts));
    expect_eq("mmproj",  plan.mmproj.path, p.mmproj);
    expect_eq("mtp",     plan.mtp.path,    p.mtp);
    expect_eq("dflash",  plan.dflash.path, p.dflash);
    expect_eq("eagle3",  plan.eagle3.path, p.eagle3);
    expect_eq("dspark",  plan.dspark.path, p.dspark);
}

// the listing reads the cache, so the repo is materialized there as a snapshot;
// a path the filesystem refuses (MAX_PATH on Windows) leaves the listing unchecked
static void check_real_listing(const real_repo & r, const std::vector<std::string> & files) {
    g_context = std::string(r.id) + " listing";

    std::string folder = r.id;
    string_replace_all(folder, "/", "--");
    const std::filesystem::path repo_dir = cache_dir / ("models--" + folder);
    std::filesystem::create_directories(repo_dir / "refs");
    { std::ofstream(repo_dir / "refs" / "main") << COMMIT; }
    for (const auto & f : files) {
        std::filesystem::path p = cached(r.id, f);
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
        { std::ofstream(p) << "gguf"; }
        if (!std::filesystem::exists(p, ec)) {
            printf("  %s: listing unchecked, the filesystem refuses %s\n", r.id, f.c_str());
            std::filesystem::remove_all(repo_dir, ec);
            return;
        }
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
    expect_eq("tags", string_join(actual, " "), string_join(expected, " "));

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
    for (const auto & c : token_cases) {
        g_repos[c.repo] = { "model-Q4_0.gguf" };
    }
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

    test_token_forwarding();
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
