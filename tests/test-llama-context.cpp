//
// test-llama-context - exercises the state handling of llama_context
//

#include "llama-context/test-case.h"
#include "llama-context/test-log.h"
#include "llama-context/test-runner.h"
#include "llama-context/test-util.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama-cpp.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// the state file of a model, named after it so that models running in parallel do not collide
static std::string test_state_path(const std::string & model_name) {
    return "dump_state." + model_name + ".bin";
}

// --name VALUE or --name=VALUE: 0 when this argument is not the option, 1 when it took a value,
//   -1 when the value is missing or empty
static int test_opt(char ** argv, int & i, int argc, const char * name, std::string & value) {
    const size_t len = std::strlen(name);

    if (std::strncmp(argv[i], name, len) != 0) {
        return 0;
    }

    if (argv[i][len] == '=') {
        value = argv[i] + len + 1;
        return value.empty() ? -1 : 1;
    }

    if (argv[i][len] != '\0') {
        return 0;
    }

    if (i + 1 >= argc) {
        return -1;
    }

    value = argv[++i];
    return value.empty() ? -1 : 1;
}

// tokenize the prompt, or generate random tokens for models without a tokenizer
static llama_tokens prepare_tokens(llama_model * model, const common_params & params) {
    llama_tokens tokens;

    if (params.prompt.empty()) {
        // the prompt must not exceed n_batch and n_ctx of cases
        constexpr int n_prompt_max = 100;
        const int n_prompt = std::min(params.n_batch, n_prompt_max);

        // this path is useful for model files that do not have a tokenizer
        TLOG_INF("%s: no prompt provided, generating %d random tokens\n", __func__, n_prompt);

        const auto * vocab = llama_model_get_vocab(model);
        const auto   n_vocab = llama_vocab_n_tokens(vocab);

        std::mt19937 rng(params.sampling.seed);
        std::uniform_int_distribution<llama_token> dist(0, n_vocab - 1);
        for (int i = 0; i < n_prompt; i++) {
            tokens.push_back(dist(rng));
        }
    } else {
        TLOG_INF("%s: tokenizing prompt '%s'\n", __func__, params.prompt.c_str());

        auto ctx = llama_context_ptr{llama_init_from_model(model, common_context_params_to_llama(params))};
        tokens = common_tokenize(ctx.get(), params.prompt, true);
    }

    TLOG_INF("%s: the input prompt is %d tokens\n", __func__, (int) tokens.size());

    return tokens;
}

static std::vector<std::string> collect_models(const std::string & models_dir) {
    std::vector<std::string> models;

    for (const auto & entry : std::filesystem::directory_iterator(models_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
            models.push_back(entry.path().string());
        }
    }
    std::sort(models.begin(), models.end());

    return models;
}

// one row of the result table
struct model_result {
    std::string name;
    std::string log_path;
    std::vector<case_run> runs;
    bool failed = false;
};

// run the selected cases for one model and return one verdict per case
//
// safe to call from a worker thread: the log capture and the state file are either per thread or
// named after the model
//
// report prints the verdict of every case as it finishes, capture_log keeps the log of the run in
// a file - table mode always captures, so that a failing model can be dumped after the table;
// single model mode only does when --logs asks for it
static model_result run_model(const std::string & model_path,
                              const common_params & params,
                              const std::vector<const ctx_test *> & cases,
                              const std::string & logs_dir,
                              bool report,
                              bool capture_log) {
    model_result res;

    res.name     = std::filesystem::path(model_path).filename().string();
    res.log_path = test_log_path(logs_dir, res.name);

    if (capture_log) {
        // capture the model from the moment it is loaded, so that even a load failure has a log
        test_log_capture(res.log_path);
    }

    struct common_params model_params = params;
    model_params.model.path = model_path;

    // a model that cannot be loaded is a failure of every case, not a skip - the cases never
    //   run, so their verdicts are recorded here
    auto llama_init = common_init_from_params(model_params, true);
    if (llama_init->model() == nullptr) {
        TLOG_ERR("%s: failed to init model '%s'\n", __func__, model_path.c_str());

        for (const auto * tc : cases) {
            case_run run;
            run.name   = tc->name();
            run.status = test_status::FAIL;
            run.reason = "the model failed to load";
            res.runs.push_back(std::move(run));
        }
    } else {
        GGML_ASSERT(llama_init->context() == nullptr);

        model_run mr;
        mr.model  = llama_init->model();
        mr.name   = res.name;
        // the cases share a working directory, so state files are named after the model
        mr.params          = std::move(model_params);
        mr.params.out_file = test_state_path(res.name);
        mr.tokens          = prepare_tokens(mr.model, mr.params);

        res.runs = run_cases(mr, cases, report);
    }

    if (capture_log) {
        test_log_capture("");
    }

    std::remove(test_state_path(res.name).c_str()); // the cases leave the session file behind

    res.failed = std::any_of(res.runs.begin(), res.runs.end(),
            [](const case_run & run) { return run.status == test_status::FAIL; });

    return res;
}

static void print_usage(int /* argc */, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  %s -m your_model.gguf\n", argv[0]);
    LOG("\n  %s --models tests/test-models\n", argv[0]);
    LOG("\n  %s --models tests/test-models --test 'rollback|replay'\n", argv[0]);
    LOG("\n  %s --models tests/test-models -j 8\n", argv[0]);
    LOG("\n  %s -m your_model.gguf -lv 5\n", argv[0]);
    LOG("\nspecial flags:\n");
    LOG("\n  --models DIR              run every .gguf in DIR and print a result table\n");
    LOG("\n  --test REGEX              run only the cases whose name matches REGEX\n");
    LOG("\n  --list                    list the registered cases and exit\n");
    LOG("\n  -j N, --threads N         run N models at a time in --models mode (default: min(nproc, 4))\n");
    LOG("\n  --logs DIR                keep the log of each model in DIR instead of discarding it\n");
    LOG("\n  --logs-on-fail always     after the table, print the log of every failed model (default)\n");
    LOG("\n  --logs-on-fail never      never print the logs of the failed models\n");
    LOG("\n");
}

static void print_cases() {
    printf("\ncases:\n");

    for (const auto & tc : test_cases()) {
        printf("  %-11s %-5s %s\n", tc->name(), tc->group() == TEST_GROUP_RS ? "[rs]" : "[kv]", tc->description());
    }

    printf("\nthe [rs] cases are also run against a cache pre-filled with a known pattern\n\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "";
    params.out_file = "dump_state.bin";
    params.sampling.seed = 1234;

    common_init();

    // extract our own options before handing the rest to the common parser
    std::string models_dir;
    std::string logs_dir;
    std::string test_filter;

    bool list_cases        = false;
    bool dump_logs_on_fail = true;

    int parallel_workers = 0; // 0 = auto, same flag as in test-backend-ops

    auto parse_parallel_workers = [&](const char * flag, const char * value) {
        try {
            parallel_workers = std::stoi(value);
        } catch (const std::exception &) {
            fprintf(stderr, "%s: expected a number, got '%s'\n", flag, value);
            return false;
        }
        if (parallel_workers < 1) {
            fprintf(stderr, "%s: expected a positive number, got %d\n", flag, parallel_workers);
            return false;
        }

        return true;
    };

    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string value;

        const int res_models = test_opt(argv, i, argc, "--models", value);
        if (res_models != 0) {
            if (res_models < 0) {
                fprintf(stderr, "--models requires a directory argument\n");
                return 1;
            }
            models_dir = value;
            continue;
        }

        const int res_logs = test_opt(argv, i, argc, "--logs", value);
        if (res_logs != 0) {
            if (res_logs < 0) {
                fprintf(stderr, "--logs requires a directory argument\n");
                return 1;
            }
            logs_dir = value;
            continue;
        }

        const int res_test = test_opt(argv, i, argc, "--test", value);
        if (res_test != 0) {
            if (res_test < 0) {
                fprintf(stderr, "--test requires a regular expression argument\n");
                return 1;
            }
            test_filter = value;
            continue;
        }

        if (test_opt(argv, i, argc, "--logs-on-fail", value) != 0) {
            if (value == "always") {
                dump_logs_on_fail = true;
            } else if (value == "never") {
                dump_logs_on_fail = false;
            } else {
                fprintf(stderr, "--logs-on-fail: expected 'always' or 'never', got '%s'\n", value.c_str());
                return 1;
            }
            continue;
        }

        // -j is the flag of test-backend-ops, --threads is its long form
        if (std::strcmp(argv[i], "-j") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "-j requires a number argument\n");
                return 1;
            }
            if (!parse_parallel_workers("-j", argv[i])) {
                return 1;
            }
            continue;
        }

        const int res_threads = test_opt(argv, i, argc, "--threads", value);
        if (res_threads != 0) {
            if (res_threads < 0) {
                fprintf(stderr, "--threads requires a number argument\n");
                return 1;
            }
            if (!parse_parallel_workers("--threads", value.c_str())) {
                return 1;
            }
            continue;
        }

        if (std::strcmp(argv[i], "--list") == 0) {
            list_cases = true;
            continue;
        }

        filtered_argv.push_back(argv[i]);
    }
    filtered_argv.push_back(nullptr);
    const int fargc = (int) filtered_argv.size() - 1;

    if (list_cases) {
        print_cases();
        return 0;
    }

    // in --models mode there is no single model; set a placeholder so the common parser's
    // "--model is required" check passes (each model is set individually inside the loop)
    if (!models_dir.empty()) {
        params.model.path = models_dir;
    }

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON, print_usage)) {
        return 1;
    }

    // one model per thread: fitting the params to the free memory would swap in a log callback
    //   that points to the stack of the thread doing the fit, which is only safe while no other
    //   thread is inside llama - and the models of this test are far too small to need it
    params.fit_params = false;

    const std::vector<const ctx_test *> cases = select_cases(test_filter);
    if (cases.empty()) {
        fprintf(stderr, "--test '%s' selected no cases, see --list\n", test_filter.c_str());
        return 1;
    }

    if (params.n_predict < 0) {
        params.n_predict = 16;
    }

    // capture the logs of each model to a file, so that they can be dumped when a model fails:
    // the table stays clean, but the details of a failure stay close at hand
    if (!logs_dir.empty() && !std::filesystem::create_directories(logs_dir)) {
        fprintf(stderr, "failed to create the log directory '%s'\n", logs_dir.c_str());
        return 1;
    }

    // route the log of every thread to its own capture - without a capture of its own, a thread
    //   writes to the console as usual
    llama_log_set(test_log_callback, nullptr);

    llama_backend_init();

    // run the full suite for a single model
    if (models_dir.empty()) {
        const int32_t n_threads = params.cpuparams.n_threads;
        LOG("threads: %d\n", n_threads);

        const model_result res = run_model(params.model.path, params, cases, logs_dir,
                true /* report */, !logs_dir.empty());

        size_t n_pass = 0;
        size_t n_skip = 0;
        size_t n_fail = 0;

        for (const auto & run : res.runs) {
            switch (run.status) {
                case test_status::PASS: n_pass++; break;
                case test_status::SKIP: n_skip++; break;
                case test_status::FAIL: n_fail++; break;
            }
        }

        fprintf(stderr, "\ncases    : %zu\n", res.runs.size());
        fprintf(stderr, "passed   : %zu\n", n_pass);
        fprintf(stderr, "skipped  : %zu\n", n_skip);
        fprintf(stderr, "failed   : %zu\n", n_fail);

        return n_fail == 0 ? 0 : 1;
    }

    // table mode: run the suite for every model in the directory
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
        TLOG_ERR("%s: models directory '%s' does not exist\n", __func__, models_dir.c_str());
        return 1;
    }

    const auto models = collect_models(models_dir);
    if (models.empty()) {
        TLOG_ERR("%s: no .gguf models found in '%s'\n", __func__, models_dir.c_str());
        return 1;
    }

    const unsigned n_hw_2 = std::thread::hardware_concurrency() / 2;
    const size_t   n_auto = std::max<size_t>(1, std::min<size_t>(n_hw_2, 4));
    const size_t   n_pool = parallel_workers > 0
        ? std::min<size_t>((size_t) parallel_workers, models.size())
        : n_auto;

    const int32_t n_threads = std::max<int32_t>(1, n_hw_2 / n_pool);

    // a column is as wide as its case name, but never narrower than the widest verdict
    const size_t verdict_width = std::max({ test_status_width(test_status::PASS),
                                            test_status_width(test_status::FAIL),
                                            test_status_width(test_status::SKIP) });

    auto col_width = [&](const char * name) { return (int) std::max(strlen(name), verdict_width); };

    size_t name_width = 5; // "Model"
    for (const auto & model_path : models) {
        name_width = std::max(name_width, std::filesystem::path(model_path).filename().string().size());
    }

    // silence everything but the table itself (LOG has verbosity LOG_LEVEL_OUTPUT = 0)
    common_log_set_verbosity_thold(0);

    if (!test_filter.empty()) {
        LOG("filter: %s\n", test_filter.c_str());
    }
    LOG("models in parallel: %zu, threads per model: %d\n", n_pool, n_threads);
    common_log_flush(common_log_main());

    LOG("%-*s", (int) name_width, "Model");
    for (const auto & tc : cases) {
        LOG("  %-*s", col_width(tc->name()), tc->name());
    }
    LOG("\n");
    common_log_flush(common_log_main());

    std::vector<size_t> n_pass(cases.size(), 0);
    std::vector<size_t> n_skip(cases.size(), 0);
    std::vector<size_t> n_fail(cases.size(), 0);

    size_t n_model_fail = 0;

    // the models are independent of each other, so they run on several threads - but the table
    //   is printed in model order, with the log of a failed model right after its row, exactly as
    //   if the models had run one by one: a worker stores its result and then prints every row
    //   that became printable in order, under the print lock
    std::vector<model_result> results(models.size());
    std::vector<char>         ready(models.size(), 0);

    std::mutex pool_mutex;
    size_t     next_print = 0;

    auto print_ready = [&]() {
        std::lock_guard<std::mutex> lock(pool_mutex);

        while (next_print < results.size() && ready[next_print]) {
            const model_result & res = results[next_print];

            LOG("%-*s", (int) name_width, res.name.c_str());

            for (size_t i = 0; i < res.runs.size(); i++) {
                const int pad = (int) (col_width(cases[i]->name()) - test_status_width(res.runs[i].status));

                LOG("  %s%*s", test_status_str(res.runs[i].status), pad, "");

                switch (res.runs[i].status) {
                    case test_status::PASS: n_pass[i]++; break;
                    case test_status::FAIL: n_fail[i]++; break;
                    case test_status::SKIP: n_skip[i]++; break;
                }
            }
            LOG("\n");
            common_log_flush(common_log_main());

            // a log is discarded when it was printed in full and nobody asked to keep it
            bool keep_log = !logs_dir.empty();

            if (res.failed) {
                n_model_fail++;

                // dump the log of a failed model once its row is printed - interleaving it with
                //   the rows would make the table unreadable
                if (dump_logs_on_fail) {
                    LOG("\n=== logs: %s (%s) ===\n", res.name.c_str(), res.log_path.c_str());
                    common_log_flush(common_log_main());

                    // a dump that had to be cut short leaves the file behind, the table cannot
                    //   show the whole log
                    keep_log = keep_log || !test_log_dump(res.log_path);
                }
            }

            if (!keep_log) {
                std::remove(res.log_path.c_str());
            }

            ++next_print;
        }
    };

    std::atomic<size_t> next_model(0);

    auto worker = [&]() {
        // no verdict lines here - the table carries the verdicts, the reasons are logged to the
        //   file of the model
        for (size_t i = next_model.fetch_add(1); i < results.size(); i = next_model.fetch_add(1)) {
            results[i] = run_model(models[i], params, cases, logs_dir,
                    false /* report */, true /* capture_log */);

            {
                std::lock_guard<std::mutex> lock(pool_mutex);
                ready[i] = 1;
            }

            print_ready();
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(n_pool > 0 ? n_pool - 1 : 0);

    for (size_t i = 1; i < n_pool; i++) {
        pool.emplace_back(worker);
    }

    worker(); // the main thread runs models like any other worker

    for (auto & t : pool) {
        t.join();
    }

    print_ready(); // the last worker can have stored its result while the others were joining

    common_log_set_verbosity_thold(LOG_DEFAULT_LLAMA);
    common_log_flush(common_log_main());

    for (size_t i = 0; i < cases.size(); i++) {
        if (n_fail[i] == 0 && n_skip[i] == 0) {
            continue;
        }
        LOG_INF("%s: %-13s %zu passed, %zu skipped, %zu failed (of %zu)\n",
                __func__, cases[i]->name(), n_pass[i], n_skip[i], n_fail[i], models.size());
    }
    LOG_INF("%s: models: %zu passed, %zu failed (of %zu)\n", __func__, models.size() - n_model_fail, n_model_fail, models.size());

    return n_model_fail == 0 ? 0 : 1;
}
