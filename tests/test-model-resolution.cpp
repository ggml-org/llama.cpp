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

//
// end-to-end assembly: real CLI parsing, real handler init resolving over the
// loopback, downloads skipped by flipping offline before apply
//

static void assemble(std::vector<std::string> argv, common_params & params) {
    std::vector<char *> cargv;
    g_context.clear();
    for (auto & a : argv) {
        g_context += g_context.empty() ? a : " " + a;
        cargv.push_back(a.data());
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
    test_task_assembly();

    server.stop();
    server_thread.join();

    std::filesystem::remove_all(cache_dir);
    printf("test-model-resolution: all tests OK\n");
    return 0;
}
