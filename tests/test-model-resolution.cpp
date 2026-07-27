// tests the HF resolution and the model handler assembly of download.cpp and
// arg.cpp by including the sources inside a namespace, with hf_cache monkey
// patched to serve synthetic listings, so the tested code is not modified and
// no network access is needed

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

#include <cassert>

#undef NDEBUG

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

// the listing served by the monkey patched hf_cache below
static ::hf_cache::hf_files g_files;

namespace hf_cache {
    using ::hf_cache::hf_file;
    using ::hf_cache::hf_files;

    static hf_files get_repo_files(const std::string & /*repo_id*/, const std::string & /*token*/) {
        return g_files;
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

// re-declare with its default argument, lost in the rename of the declaration
int common_download_file_single(const std::string & url, const std::string & path, const common_download_opts & opts, bool is_hf = true);

// unqualified hf_cache:: lookups inside the sources now bind to the fakes above
#include "download.cpp"
#include "arg.cpp"

} // namespace dl_test

// build a synthetic listing from repo-relative paths
static hf_cache::hf_files repo(const std::vector<std::string> & paths) {
    hf_cache::hf_files files;
    for (const auto & p : paths) {
        hf_cache::hf_file f;
        f.path       = p;
        f.url        = "https://hf.test/repo/resolve/main/" + p;
        f.local_path = "/tmp/hf-test/" + p;
        f.repo_id    = "test/repo";
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

//
// static helpers, reached directly through the namespace include
//

static void test_statics() {
    printf("test-model-resolution: static helpers\n");

    // quant bits extraction on real world namings
    assert(dl_test::extract_quant_bits("model-Q8_0.gguf")                   == 8);
    assert(dl_test::extract_quant_bits("model-UD-Q2_K_XL.gguf")             == 2);
    assert(dl_test::extract_quant_bits("model.i1-Q6_K.gguf")                == 6);
    assert(dl_test::extract_quant_bits("model-BF16.gguf")                   == 16);
    assert(dl_test::extract_quant_bits("model-MXFP4-00001-of-00002.gguf")   == 4);

    // sidecar keywords are never a model, wherever they appear in the name
    assert(dl_test::gguf_filename_is_model("model-Q8_0.gguf"));
    assert(!dl_test::gguf_filename_is_model("mtp-model-Q8_0.gguf"));
    assert(!dl_test::gguf_filename_is_model("model-mtp-Q8_0.gguf"));
    assert(!dl_test::gguf_filename_is_model("dflash-model-Q8_0.gguf"));
    assert(!dl_test::gguf_filename_is_model("eagle3-model-Q8_0.gguf"));
    assert(!dl_test::gguf_filename_is_model("mmproj-model-Q8_0.gguf"));
    assert(!dl_test::gguf_filename_is_model("model.txt"));

    // the sibling picker honors an exact tag over quant proximity
    auto files = repo({"mtp-model-BF16.gguf", "mtp-model-Q4_0.gguf", "mtp-model-Q8_0.gguf"});
    assert(dl_test::find_best_sibling(files, "model-Q8_0.gguf", "mtp-", "Q4_0").path == "mtp-model-Q4_0.gguf");
    assert(dl_test::find_best_sibling(files, "model-Q8_0.gguf", "mtp-").path        == "mtp-model-Q8_0.gguf");
}

//
// table-driven plan resolution through the real entry point,
// each case replayed on permutations of the listing to assert determinism
//

struct plan_case {
    const char * name;
    const std::vector<std::string> & files;
    const char * hf_repo;
    const char * hf_file;
    bool sidecars; // request mmproj + mtp + dflash + eagle3
    const char * primary;
    size_t       model_files;
    const char * mmproj;
    const char * mtp;
    const char * dflash;
};

static const plan_case plan_cases[] = {
    // exact tag picks the matching primary, sidecars follow the tag
    {"flat exact tag",       flat,    "test/repo:Q8_0",   "", true,  "model-Q8_0.gguf",   1, "mmproj-model-Q8_0.gguf", "mtp-model-Q8_0.gguf",  "dflash-model-Q8_0.gguf"},
    // no tag falls back to the default preference
    {"flat default",         flat,    "test/repo",        "", false, "model-Q4_K_M.gguf", 1, "",                       "",                     ""},
    // no tag and no default match falls back to the first model, never a sidecar
    {"unsloth fallback",     unsloth, "test/repo",        "", true,  "model-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf",  "",                     ""},
    // explicit hf_file picks that exact file
    {"flat hf_file",         flat,    "test/repo",        "model-BF16.gguf", false, "model-BF16.gguf", 1, "",         "",                     ""},
    // missing hf_file resolves nothing
    {"flat missing hf_file", flat,    "test/repo",        "nope.gguf", false, "",                0, "",               "",                     ""},
    // a sharded primary brings all its parts, a subdir primary finds the root sidecar
    {"subdir shards",        subdir,  "test/repo:Q3_K_M", "", true,  "Q3_K_M/model-Q3_K_M-00001-of-00003.gguf", 3, "mmproj-model-f16.gguf", "model-mtp-Q8_0.gguf", ""},
    // a tag with no matching full model still resolves the requested sidecar
    {"hole tag sidecar",     hole,    "test/repo:Q4_0",   "", true,  "",                  0, "",                       "mtp-model-Q4_0.gguf",  "dflash-model-Q8_0.gguf"},
    // the same tag without a requested sidecar resolves nothing
    {"hole tag alone",       hole,    "test/repo:Q4_0",   "", false, "",                  0, "",                       "",                     ""},
    // no tag anchors the sidecar on the primary quant
    {"hole default anchor",  hole,    "test/repo",        "", true,  "model-Q4_K_M.gguf", 1, "",                       "mtp-model-Q4_0.gguf",  ""},
    // the mtp- keyword is case sensitive, a suffix -MTP file is not discovered
    {"unsloth suffix mtp",   unsloth, "test/repo:Q8_K_XL", "", true, "model-UD-Q8_K_XL.gguf", 1, "mmproj-BF16.gguf",  "",                     ""},
    // vendor prefixes and the dot quant convention both match the tag
    {"vendor prefix",        vendors, "test/repo:Q8_0",   "", false, "TheDrummer_Model-24B-v4.1-Q8_0.gguf", 1, "",   "",                     ""},
};

static void check_plan(const plan_case & c) {
    common_download_opts opts;
    opts.download_mmproj = c.sidecars;
    opts.download_mtp    = c.sidecars;
    opts.download_dflash = c.sidecars;
    opts.download_eagle3 = c.sidecars;

    auto plan = dl_test::common_download_get_hf_plan(model_ref(c.hf_repo, c.hf_file), opts);

    assert(plan.primary.path         == c.primary);
    assert(plan.model_files.size()   == c.model_files);
    assert(plan.mmproj.path          == c.mmproj);
    assert(plan.mtp.path             == c.mtp);
    assert(plan.dflash.path          == c.dflash);

    // invariant: the primary is never a sidecar file
    if (!plan.primary.path.empty()) {
        assert(dl_test::gguf_filename_is_model(plan.primary.path));
    }
}

static void test_plan_resolution() {
    printf("test-model-resolution: plan resolution on %zu cases\n", sizeof(plan_cases) / sizeof(plan_cases[0]));

    for (const auto & c : plan_cases) {
        auto files = repo(c.files);

        // invariant: the resolution is insensitive to the listing order
        for (size_t rot = 0; rot < files.size(); ++rot) {
            dl_test::g_files = files;
            std::rotate(dl_test::g_files.begin(), dl_test::g_files.begin() + rot, dl_test::g_files.end());
            if (rot % 2 == 1) {
                std::reverse(dl_test::g_files.begin(), dl_test::g_files.end());
            }
            // the no-tag default resolution is the documented exception: it
            // legitimately depends on the listing order for its fallback pick
            if (std::string(c.hf_repo).find(':') == std::string::npos && rot > 0) {
                continue;
            }
            check_plan(c);
        }
    }
}

//
// end-to-end assembly: real CLI parsing, real handler, monkey patched listing,
// downloads skipped by flipping offline between init and apply
//

static void assemble(std::vector<std::string> argv,
                     ::common_params & params,
                     const std::vector<std::string> & files_main,
                     const std::vector<std::string> & files_spec) {
    std::vector<char *> cargv;
    for (auto & a : argv) {
        cargv.push_back(const_cast<char *>(a.c_str()));
    }
    bool ok = dl_test::common_params_parse((int) cargv.size(), cargv.data(), params, LLAMA_EXAMPLE_SERVER);
    assert(ok);

    dl_test::g_files = repo(files_main);
    auto handler = dl_test::common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
    if (!params.speculative.draft.mparams.hf_repo.empty()) {
        // the draft plan resolves against its own repo listing
        dl_test::g_files = repo(files_spec);
        handler.plan_spec = dl_test::common_download_get_hf_plan(params.speculative.draft.mparams, handler.opts);
    }

    // skip the network execution, on_done still wires the params
    params.offline = true;
    dl_test::common_models_handler_apply(handler, params);
}

static void test_task_assembly() {
    printf("test-model-resolution: end-to-end assembly\n");

    {
        // plain -hf wires the model and its mmproj, nothing speculative
        ::common_params params;
        assemble({"server", "-hf", "test/repo:Q8_0"}, params, flat, {});
        assert(params.model.path  == "/tmp/hf-test/model-Q8_0.gguf");
        assert(params.mmproj.path == "/tmp/hf-test/mmproj-model-Q8_0.gguf");
        assert(params.speculative.draft.mparams.path.empty());
    }
    {
        // -hf with a spec type wires the sidecar of the main repo as fallback draft
        ::common_params params;
        assemble({"server", "-hf", "test/repo:Q8_0", "--spec-type", "draft-mtp"}, params, flat, {});
        assert(params.speculative.draft.mparams.path == "/tmp/hf-test/mtp-model-Q8_0.gguf");
    }
    {
        // -hfd with a spec type wires the draft repo sidecar at its tag,
        // not its full model, and suppresses the main repo fallback
        ::common_params params;
        assemble({"server", "-hf", "test/repo:Q8_0", "-hfd", "test/repo:Q4_0", "--spec-type", "draft-mtp"}, params, hole, hole);
        assert(params.speculative.draft.mparams.path == "/tmp/hf-test/mtp-model-Q4_0.gguf");
    }
    {
        // an explicit -md file wins over the sidecar resolution
        ::common_params params;
        assemble({"server", "-hf", "test/repo:Q8_0", "-hfd", "test/repo", "-md", "mtp-model-BF16.gguf", "--spec-type", "draft-mtp"}, params, flat, flat);
        assert(params.speculative.draft.mparams.path == "/tmp/hf-test/mtp-model-BF16.gguf");
    }
    {
        // -hfd without a spec type auto-selects the type from the shipped sidecars, mtp first
        ::common_params params;
        assemble({"server", "-hf", "test/repo:Q8_0", "-hfd", "test/repo:Q8_0"}, params, flat, flat);
        assert(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP});
        assert(params.speculative.draft.mparams.path == "/tmp/hf-test/mtp-model-Q8_0.gguf");
    }
    {
        // -hfd on a repo without sidecars keeps resolving a full model as draft
        ::common_params params;
        assemble({"server", "-hf", "test/repo:Q8_0", "-hfd", "test/draft"}, params, flat, {"draft-model-Q4_K_M.gguf"});
        assert(params.speculative.types == std::vector<enum common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE});
        assert(params.speculative.draft.mparams.path == "/tmp/hf-test/draft-model-Q4_K_M.gguf");
    }
    {
        // a preset repo wires the preset and clears the model for router mode
        ::common_params params;
        assemble({"server", "-hf", "test/repo"}, params, {"preset.ini", "model-Q8_0.gguf"}, {});
        assert(params.models_preset == "/tmp/hf-test/preset.ini");
        assert(params.model.path.empty());
        assert(params.model.hf_repo.empty());
    }
}

int main(void) {
    test_statics();
    test_plan_resolution();
    test_task_assembly();
    printf("test-model-resolution: all tests OK\n");
    return 0;
}
