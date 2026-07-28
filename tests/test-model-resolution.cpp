// tests the HF resolution and the model handler assembly of download.cpp and
// arg.cpp by including the sources inside a namespace, with hf_cache monkey
// patched to serve synthetic listings, so the tested code is not modified and
// no network access is needed

// this TU compiles a private namespaced copy of the sources, so many of their
// static functions and the statics of transitively included headers are
// legitimately unused here
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#endif

// arg.cpp includes windows headers on _WIN32, pre-include them here so they
// stay outside the namespace below
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
#include <shellapi.h>
#endif

// pre-included so their include guards keep these global inside the namespace,
// arg.h and preset.h are NOT pre-included: the sources pull them inside the
// namespace below, so their declarations and definitions live together
#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "hf-cache.h"
#include "download.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "http.h"

// independent of NDEBUG, so the checks stay alive in Release builds
#define REQUIRE(x) do {                                                       \
    if (!(x)) {                                                               \
        fprintf(stderr, "%s:%d: REQUIRE(%s) failed\n", __FILE__, __LINE__, #x); \
        std::abort();                                                         \
    }                                                                         \
} while (0)

// download.h is pre-included above, so its declarations stay global; rename
// every function it declares for the whole TU, otherwise ADL on their global
// argument types would make the namespaced definitions ambiguous with them
#define common_docker_resolve_model    dl_common_docker_resolve_model
#define common_download_file_single    dl_common_download_file_single
#define common_download_get_all_parts  dl_common_download_get_all_parts
#define common_download_get_hf_plan    dl_common_download_get_hf_plan
#define common_download_remove         dl_common_download_remove
#define common_download_run_tasks      dl_common_download_run_tasks
#define common_download_split_repo_tag dl_common_download_split_repo_tag
#define common_list_cached_models      dl_common_list_cached_models
#define common_remote_get_content      dl_common_remote_get_content

namespace dl_test {

// the listings served by the monkey patched hf_cache below, keyed by repo id,
// so common_models_handler_init resolves its main, draft and vocoder plans
// exactly as in production
static std::map<std::string, ::hf_cache::hf_files> g_repos;

namespace hf_cache {
    using ::hf_cache::hf_file;
    using ::hf_cache::hf_files;

    static hf_files get_repo_files(const std::string & repo_id, const std::string & /*token*/) {
        auto it = g_repos.find(repo_id);
        return it != g_repos.end() ? it->second : hf_files{};
    }

    static hf_files get_cached_files(const std::string & /*repo_id*/ = {}) {
        return {};
    }

    static bool remove_cached_repo(const std::string & /*repo_id*/) {
        return false;
    }

    static std::string finalize_file(const hf_file & file) {
        return file.local_path;
    }
} // namespace hf_cache

// previous declarations for the renamed copies of the download.h functions,
// written with their original names so the macros above keep them in sync,
// with the default argument of the download one restored
std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);
void common_download_run_tasks(const std::vector<common_download_task> & tasks);
std::vector<std::string> common_download_get_all_parts(const std::string & url);
std::vector<common_cached_model_info> common_list_cached_models();
int common_download_file_single(const std::string & url, const std::string & path, const common_download_opts & opts = {}, bool skip_etag = false);
std::string common_docker_resolve_model(const std::string & docker);
bool common_download_remove(const std::string & hf_repo_with_tag);
common_download_hf_plan common_download_get_hf_plan(const common_params_model & model, const common_download_opts & opts);

// unqualified hf_cache:: lookups inside the sources now bind to the fakes above
#include "download.cpp"
#include "arg.cpp"

} // namespace dl_test

// build a synthetic listing from repo-relative paths
static hf_cache::hf_files repo(const std::string & repo_id, const std::vector<std::string> & paths) {
    hf_cache::hf_files files;
    for (const auto & p : paths) {
        hf_cache::hf_file f;
        f.path       = p;
        f.url        = "https://hf.test/" + repo_id + "/resolve/main/" + p;
        f.local_path = "/tmp/hf-test/" + repo_id + "/" + p;
        f.repo_id    = repo_id;
        files.push_back(f);
    }
    return files;
}

static common_params_model model_ref(const std::string & hf_repo, const std::string & hf_file = "") {
    common_params_model m;
    m.hf_repo = hf_repo;
    m.hf_file = hf_file;
    return m;
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
// static helpers, reached directly through the namespace include
//

static void test_statics() {
    printf("test-model-resolution: static helpers\n");

    // quant bits extraction on real world namings
    REQUIRE(dl_test::extract_quant_bits("model-Q8_0.gguf")                 == 8);
    REQUIRE(dl_test::extract_quant_bits("model-UD-Q2_K_XL.gguf")           == 2);
    REQUIRE(dl_test::extract_quant_bits("model.i1-Q6_K.gguf")              == 6);
    REQUIRE(dl_test::extract_quant_bits("model-BF16.gguf")                 == 16);
    REQUIRE(dl_test::extract_quant_bits("model-MXFP4-00001-of-00002.gguf") == 4);

    // sidecar keywords are never a model, wherever they appear in the name
    REQUIRE(dl_test::gguf_filename_is_model("model-Q8_0.gguf"));
    REQUIRE(!dl_test::gguf_filename_is_model("mtp-model-Q8_0.gguf"));
    REQUIRE(!dl_test::gguf_filename_is_model("model-mtp-Q8_0.gguf"));
    REQUIRE(!dl_test::gguf_filename_is_model("dflash-model-Q8_0.gguf"));
    REQUIRE(!dl_test::gguf_filename_is_model("eagle3-model-Q8_0.gguf"));
    REQUIRE(!dl_test::gguf_filename_is_model("mmproj-model-Q8_0.gguf"));
    REQUIRE(!dl_test::gguf_filename_is_model("model.txt"));

    // the sibling picker honors an exact tag over quant proximity
    auto files = repo("test/repo", {"mtp-model-BF16.gguf", "mtp-model-Q4_0.gguf", "mtp-model-Q8_0.gguf"});
    REQUIRE(dl_test::find_best_sibling(files, "model-Q8_0.gguf", "mtp-", "Q4_0").path == "mtp-model-Q4_0.gguf");
    REQUIRE(dl_test::find_best_sibling(files, "model-Q8_0.gguf", "mtp-").path        == "mtp-model-Q8_0.gguf");
}

//
// table-driven plan resolution through the real entry point,
// each case replayed on permutations of the listing to assert determinism,
// except the cases that legitimately depend on the listing order
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

    auto plan = dl_test::common_download_get_hf_plan(model_ref(c.hf_repo, c.hf_file), opts);

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
        REQUIRE(dl_test::gguf_filename_is_model(plan.primary.path));
    }
}

static void test_plan_resolution() {
    printf("test-model-resolution: plan resolution on %zu cases\n", sizeof(plan_cases) / sizeof(plan_cases[0]));

    for (const auto & c : plan_cases) {
        auto files = repo("test/repo", c.files);

        // invariant: the resolution is insensitive to the listing order
        for (size_t rot = 0; rot < files.size(); ++rot) {
            if (c.order_dependent && rot > 0) {
                continue;
            }
            dl_test::g_repos["test/repo"] = files;
            std::rotate(dl_test::g_repos["test/repo"].begin(),
                        dl_test::g_repos["test/repo"].begin() + rot,
                        dl_test::g_repos["test/repo"].end());
            if (rot % 2 == 1) {
                std::reverse(dl_test::g_repos["test/repo"].begin(), dl_test::g_repos["test/repo"].end());
            }
            check_plan(c);
        }
    }
    dl_test::g_repos.clear();
}

//
// end-to-end assembly: real CLI parsing, real handler init on the mapped fake
// cache, downloads skipped by flipping offline between init and apply
//

static void assemble(std::vector<std::string> argv, ::common_params & params) {
    std::vector<char *> cargv;
    for (auto & a : argv) {
        cargv.push_back(const_cast<char *>(a.c_str()));
    }
    bool ok = dl_test::common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_SERVER);
    REQUIRE(ok);

    auto handler = dl_test::common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);

    // skip the network execution, on_done still wires the params
    params.offline = true;
    dl_test::common_models_handler_apply(handler, params);
}

static void test_task_assembly() {
    printf("test-model-resolution: end-to-end assembly\n");

    dl_test::g_repos["test/main"]   = repo("test/main",   flat);
    dl_test::g_repos["test/hole"]   = repo("test/hole",   hole);
    dl_test::g_repos["test/trio"]   = repo("test/trio",   trio);
    dl_test::g_repos["test/dflash"] = repo("test/dflash", dflash_only);
    dl_test::g_repos["test/eagle3"] = repo("test/eagle3", eagle3_only);
    dl_test::g_repos["test/small"]  = repo("test/small",  {"draft-model-Q4_K_M.gguf"});
    dl_test::g_repos["test/preset"] = repo("test/preset", {"preset.ini", "model-Q8_0.gguf"});

    {
        // plain -hf wires the model and its mmproj, nothing speculative
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0"}, params);
        REQUIRE(params.model.path  == "/tmp/hf-test/test/main/model-Q8_0.gguf");
        REQUIRE(params.mmproj.path == "/tmp/hf-test/test/main/mmproj-model-Q8_0.gguf");
        REQUIRE(params.speculative.draft.mparams.path.empty());
    }
    {
        // --no-mmproj disables the mmproj discovery
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--no-mmproj"}, params);
        REQUIRE(params.mmproj.path.empty());
    }
    {
        // an explicit --mmproj wins over the discovery
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--mmproj", "/local/mmproj.gguf"}, params);
        REQUIRE(params.mmproj.path == "/local/mmproj.gguf");
    }
    {
        // -hf with a spec type wires the sidecar of the main repo as fallback draft
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "--spec-type", "draft-mtp"}, params);
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/main/mtp-model-Q8_0.gguf");
    }
    {
        // -hfd with a spec type wires the draft repo sidecar at its tag,
        // not its full model, and suppresses the main repo fallback
        ::common_params params;
        assemble({"server", "-hf", "test/hole:Q8_0", "-hfd", "test/hole:Q4_0", "--spec-type", "draft-mtp"}, params);
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/hole/mtp-model-Q4_0.gguf");
    }
    {
        // an explicit -md file wins over the sidecar resolution
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/main", "-md", "mtp-model-BF16.gguf", "--spec-type", "draft-mtp"}, params);
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/main/mtp-model-BF16.gguf");
    }
    {
        // -hfd without a spec type auto-selects the type, mtp first when all ship
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/trio:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP});
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/trio/mtp-model-Q8_0.gguf");
    }
    {
        // auto-selection with only a dflash sidecar
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/dflash:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH});
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/dflash/dflash-model-Q8_0.gguf");
    }
    {
        // auto-selection with only an eagle3 sidecar
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/eagle3:Q8_0"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3});
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/eagle3/eagle3-model-Q8_0.gguf");
    }
    {
        // -hfd on a repo without sidecars keeps resolving a full model as draft
        ::common_params params;
        assemble({"server", "-hf", "test/main:Q8_0", "-hfd", "test/small"}, params);
        REQUIRE(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE});
        REQUIRE(params.speculative.draft.mparams.path == "/tmp/hf-test/test/small/draft-model-Q4_K_M.gguf");
    }
    {
        // a preset repo wires the preset and clears the model for router mode
        ::common_params params;
        assemble({"server", "-hf", "test/preset"}, params);
        REQUIRE(params.models_preset == "/tmp/hf-test/test/preset/preset.ini");
        REQUIRE(params.model.path.empty());
        REQUIRE(params.model.hf_repo.empty());
    }

    dl_test::g_repos.clear();
}

int main(void) {
    test_statics();
    test_plan_resolution();
    test_task_assembly();
    printf("test-model-resolution: all tests OK\n");
    return 0;
}
