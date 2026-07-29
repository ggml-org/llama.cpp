// tests the HF model resolution and the model handler assembly end-to-end on
// synthetic repo listings: the common_http_client factory is re-assigned to a
// stub serving hardcoded HF API responses, so the real hf_cache, resolution
// and CLI parsing run against them without network access

#include "arg.h"
#include "common.h"
#include "download.h"
#include "hf-cache.h"
#include "http.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

// http.h defines file-static helpers that this TU does not call
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

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
// synthetic repos served by the stubbed HTTP client, keyed by repo id
//

static std::map<std::string, std::vector<std::string>> g_repos;

static const char * COMMIT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

struct http_client_stub : common_http_client {
    http_client_stub(const std::string & url) : common_http_client(url) {}

    httplib::Result respond(const std::string & path) {
        auto res = std::make_unique<httplib::Response>();
        res->status = 404;

        auto pos = path.find("/api/models/");
        if (pos != std::string::npos) {
            auto rest = path.substr(pos + 12);
            auto refs = rest.find("/refs");
            auto tree = rest.find("/tree/");
            if (refs != std::string::npos && g_repos.count(rest.substr(0, refs))) {
                res->status = 200;
                res->body = nlohmann::json{{"branches", {{{"name", "main"}, {"targetCommit", COMMIT}}}}}.dump();
            } else if (tree != std::string::npos && g_repos.count(rest.substr(0, tree))) {
                auto files = nlohmann::json::array();
                size_t i = 0;
                for (const auto & p : g_repos[rest.substr(0, tree)]) {
                    char oid[41];
                    snprintf(oid, sizeof(oid), "%040zx", ++i);
                    files.push_back({{"type", "file"}, {"path", p}, {"size", 1}, {"oid", oid}});
                }
                res->status = 200;
                res->body = files.dump();
            }
        }
        return httplib::Result{std::move(res), httplib::Error::Success};
    }

    httplib::Result Head(const std::string & path) override { return respond(path); }
    httplib::Result Get (const std::string & path) override { return respond(path); }
    httplib::Result Get (const std::string & path, const httplib::Headers &) override { return respond(path); }
    httplib::Result Get (const std::string & path, const httplib::Headers &, httplib::ContentReceiver, httplib::DownloadProgress) override { return respond(path); }
    httplib::Result Post(const std::string & path, const std::string &, const std::string &) override { return respond(path); }
    httplib::Result Post(const std::string & path, const httplib::Headers &, const std::string &, const std::string &, httplib::ContentReceiver) override { return respond(path); }
};

static common_params_model model_ref(const std::string & hf_repo, const std::string & hf_file = "") {
    common_params_model m;
    m.hf_repo = hf_repo;
    m.hf_file = hf_file;
    return m;
}

static bool path_ends_with(const std::string & s, const std::string & filename) {
    const std::string suffix = "/" + filename;
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
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

//
// table-driven plan resolution through the real entry point,
// each case replayed on multiple deterministic reorderings of the listing,
// except the cases whose pick legitimately depends on the listing order
//

struct plan_case {
    const char * name;
    const std::vector<std::string> & files;
    const char * hf_repo;
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
    {"flat exact tag", flat, "test/repo:Q8_0", "", true, false,
     "model-Q8_0.gguf", {"model-Q8_0.gguf"},
     "mmproj-model-Q8_0.gguf", "mtp-model-Q8_0.gguf", "dflash-model-Q8_0.gguf", ""},

    // no tag falls back to the default quant preference
    {"flat default", flat, "test/repo", "", false, false,
     "model-Q4_K_M.gguf", {"model-Q4_K_M.gguf"},
     "", "", "", ""},

    // no tag and no default match falls back to the first model in the listing
    {"unsloth fallback", unsloth, "test/repo", "", true, true,
     "model-UD-Q8_K_XL.gguf", {"model-UD-Q8_K_XL.gguf"},
     "mmproj-BF16.gguf", "", "", ""},

    // explicit hf_file picks that exact file
    {"flat hf_file", flat, "test/repo", "model-BF16.gguf", false, false,
     "model-BF16.gguf", {"model-BF16.gguf"},
     "", "", "", ""},

    // missing hf_file resolves nothing
    {"flat missing hf_file", flat, "test/repo", "nope.gguf", false, false,
     "", {},
     "", "", "", ""},

    // a sharded primary brings all its parts, a subdir primary finds the root sidecar
    {"subdir shards", subdir, "test/repo:Q3_K_M", "", true, false,
     "Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
     {"Q3_K_M/model-Q3_K_M-00001-of-00003.gguf",
      "Q3_K_M/model-Q3_K_M-00002-of-00003.gguf",
      "Q3_K_M/model-Q3_K_M-00003-of-00003.gguf"},
     "mmproj-model-f16.gguf", "model-mtp-Q8_0.gguf", "", ""},

    // a tag with no matching full model still resolves the requested sidecars
    {"hole tag sidecar", hole, "test/repo:Q4_0", "", true, false,
     "", {},
     "", "mtp-model-Q4_0.gguf", "dflash-model-Q8_0.gguf", ""},

    // the same tag without a requested sidecar resolves nothing
    {"hole tag alone", hole, "test/repo:Q4_0", "", false, false,
     "", {},
     "", "", "", ""},

    // no tag anchors the sidecars on the primary quant
    {"hole default anchor", hole, "test/repo", "", true, false,
     "model-Q4_K_M.gguf", {"model-Q4_K_M.gguf"},
     "", "mtp-model-Q4_0.gguf", "dflash-model-Q8_0.gguf", ""},

    // the mtp- keyword is case sensitive, a suffix -MTP file is not discovered
    {"unsloth suffix mtp", unsloth, "test/repo:Q8_K_XL", "", true, false,
     "model-UD-Q8_K_XL.gguf", {"model-UD-Q8_K_XL.gguf"},
     "mmproj-BF16.gguf", "", "", ""},

    // vendor prefixes and the dot quant convention both match the tag,
    // first match wins between two files at the same quant
    {"vendor prefix", vendors, "test/repo:Q8_0", "", false, true,
     "TheDrummer_Model-24B-v4.1-Q8_0.gguf", {"TheDrummer_Model-24B-v4.1-Q8_0.gguf"},
     "", "", "", ""},

    // every sidecar type resolves at the tag
    {"trio exact tag", trio, "test/repo:Q8_0", "", true, false,
     "model-Q8_0.gguf", {"model-Q8_0.gguf"},
     "", "mtp-model-Q8_0.gguf", "dflash-model-Q8_0.gguf", "eagle3-model-Q8_0.gguf"},
};

static void check_plan(const plan_case & c) {
    common_download_opts opts;
    opts.download_mmproj = c.sidecars;
    opts.download_mtp    = c.sidecars;
    opts.download_dflash = c.sidecars;
    opts.download_eagle3 = c.sidecars;

    auto plan = common_download_get_hf_plan(model_ref(c.hf_repo, c.hf_file), opts);

    REQUIRE_EQ(plan.primary.path, c.primary);
    REQUIRE_EQ(plan.mmproj.path,  c.mmproj);
    REQUIRE_EQ(plan.mtp.path,     c.mtp);
    REQUIRE_EQ(plan.dflash.path,  c.dflash);
    REQUIRE_EQ(plan.eagle3.path,  c.eagle3);

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
        // invariant: the resolution is insensitive to the listing order,
        // the cases resolving nothing are checked once to keep the logs short
        const bool empty_result = c.primary[0] == 0 && c.mtp[0] == 0 && c.dflash[0] == 0 && c.eagle3[0] == 0;
        for (size_t rot = 0; rot < c.files.size(); ++rot) {
            if ((c.order_dependent || empty_result) && rot > 0) {
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

//
// end-to-end assembly: real CLI parsing, real handler init resolving through
// the stubbed HTTP client, downloads skipped by flipping offline before apply
//

static void assemble(std::vector<std::string> argv, common_params & params) {
    std::vector<char *> cargv;
    g_context.clear();
    for (auto & a : argv) {
        g_context += g_context.empty() ? a : " " + a;
        cargv.push_back(const_cast<char *>(a.c_str()));
    }
    bool ok = common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_SERVER);
    REQUIRE(ok);

    auto handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);

    // skip the network execution, on_done still wires the params
    params.offline = true;
    common_models_handler_apply(handler, params);
}

static void test_task_assembly() {
    printf("test-model-resolution: end-to-end assembly\n");

    g_repos["test/main"]   = flat;
    g_repos["test/hole"]   = hole;
    g_repos["test/trio"]   = trio;
    g_repos["test/dflash"] = dflash_only;
    g_repos["test/eagle3"] = eagle3_only;
    g_repos["test/small"]  = {"draft-model-Q4_K_M.gguf"};
    g_repos["test/preset"] = {"preset.ini", "model-Q8_0.gguf"};

    {
        // plain -hf wires the model and its mmproj, nothing speculative
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0"}, params);
        REQUIRE(path_ends_with(params.model.path,  "model-Q8_0.gguf"));
        REQUIRE(path_ends_with(params.mmproj.path, "mmproj-model-Q8_0.gguf"));
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
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "mtp-model-Q8_0.gguf"));
    }
    {
        // -hfd with a spec type wires the draft repo sidecar at its tag,
        // not its full model, and suppresses the main repo fallback
        common_params params;
        assemble({"server", "-hf", "test/hole:Q8_0", "-hfd", "test/hole:Q4_0", "--spec-type", "draft-mtp"}, params);
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "mtp-model-Q4_0.gguf"));
    }
    {
        // an explicit -md file wins over the sidecar resolution
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/main", "-md", "mtp-model-BF16.gguf", "--spec-type", "draft-mtp"}, params);
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "mtp-model-BF16.gguf"));
    }
    {
        // -hfd without a spec type auto-selects the type, mtp first when all ship
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/trio:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP});
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "mtp-model-Q8_0.gguf"));
    }
    {
        // auto-selection with only a dflash sidecar
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/dflash:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH});
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "dflash-model-Q8_0.gguf"));
    }
    {
        // auto-selection with only an eagle3 sidecar
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/eagle3:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3});
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "eagle3-model-Q8_0.gguf"));
    }
    {
        // -hfd on a repo without sidecars keeps resolving a full model as draft
        common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/small"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE});
        REQUIRE(path_ends_with(params.speculative.draft.mparams.path, "draft-model-Q4_K_M.gguf"));
    }
    {
        // a preset repo wires the preset and clears the model for router mode
        common_params params;
        assemble({"server", "-hf", "test/preset"}, params);
        REQUIRE(path_ends_with(params.models_preset, "preset.ini"));
        REQUIRE(params.model.path.empty());
        REQUIRE(params.model.hf_repo.empty());
    }

    g_repos.clear();
}

int main(void) {
    // isolate the cache and serve every HTTP request from the stub
    std::string cache = (std::filesystem::temp_directory_path() / "test-model-resolution-cache").string();
    std::filesystem::remove_all(cache);
    setenv("LLAMA_CACHE", cache.c_str(), 1);

    common_http_client_factory = [](const std::string & url) -> common_http_client_ptr {
        return std::make_unique<http_client_stub>(url);
    };

    test_plan_resolution();
    test_task_assembly();
    printf("test-model-resolution: all tests OK\n");
    return 0;
}
