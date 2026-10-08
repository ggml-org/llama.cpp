#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace test_runner {

struct Case {
    std::string           name;
    std::function<void()> run;
    bool                  requires_device = true;
};

enum class OutputMode {
    Failed,
    All,
    None,
};

struct Config {
    std::string suite_name  = "test";
    std::string list_flag   = "--test-list";
    std::string case_flag   = "--test-case";
    std::string filter_flag = "--test-filter";
    std::string jobs_flag   = "--test-jobs";
    std::string output_flag = "--test-output";
    std::string env_jobs;
    std::string env_filter;
    std::string env_output;
    size_t      max_default_jobs = 4;

    static Config with_prefix(std::string         suite_name,
                              const std::string & flag_prefix,
                              const std::string & env_prefix,
                              size_t              max_default_jobs = 4) {
        Config config;
        config.suite_name       = std::move(suite_name);
        config.list_flag        = "--" + flag_prefix + "-test-list";
        config.case_flag        = "--" + flag_prefix + "-test-case";
        config.filter_flag      = "--" + flag_prefix + "-test-filter";
        config.jobs_flag        = "--" + flag_prefix + "-test-jobs";
        config.output_flag      = "--" + flag_prefix + "-test-output";
        config.env_jobs         = env_prefix + "_JOBS";
        config.env_filter       = env_prefix + "_FILTER";
        config.env_output       = env_prefix + "_OUTPUT";
        config.max_default_jobs = max_default_jobs;
        return config;
    }
};

struct Options {
    std::string executable;
    std::string single_case;
    std::string filter;
    size_t      jobs        = 0;
    bool        help        = false;
    bool        list        = false;
    bool        runner_control = false;
    OutputMode  output      = OutputMode::Failed;
};

struct Result {
    std::string name;
    std::string output;
    int         exit_code   = -1;
    int         signal_code = 0;
    int64_t     elapsed_ms  = 0;
    bool        passed      = false;
};

struct Selection {
    std::vector<Case> cases;
    size_t            matched        = 0;
    size_t            skipped_device = 0;
};

inline void add_case(std::vector<Case> &   cases,
                     std::string           name,
                     std::function<void()> run,
                     bool                  requires_device = true) {
    cases.push_back({ std::move(name), std::move(run), requires_device });
}

inline size_t default_jobs(const Config & config) {
    const unsigned int hardware_jobs = std::thread::hardware_concurrency();
    if (hardware_jobs == 0) {
        return 1;
    }
    return std::min<size_t>(config.max_default_jobs, hardware_jobs);
}

inline bool parse_positive_size(const char * value, size_t & out) {
    if (value == nullptr || *value == '\0') {
        return false;
    }
    char * end                 = nullptr;
    errno                      = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        return false;
    }
    out = static_cast<size_t>(parsed);
    return true;
}

inline bool parse_output_mode(const char * value, OutputMode & out) {
    if (value == nullptr) {
        return true;
    }
    const std::string mode = value;
    if (mode == "failed" || mode == "fail") {
        out = OutputMode::Failed;
        return true;
    }
    if (mode == "all") {
        out = OutputMode::All;
        return true;
    }
    if (mode == "none") {
        out = OutputMode::None;
        return true;
    }
    return false;
}

inline void print_usage(const Config & config, const char * executable) {
    std::fprintf(stderr,
                 "usage: %s [%s N] [%s REGEX] [%s MODE] [%s]\n"
                 "       %s %s NAME\n"
                 "\n"
                 "environment: %s, %s, %s\n"
                 "output modes: failed, all, none\n",
                 executable, config.jobs_flag.c_str(), config.filter_flag.c_str(), config.output_flag.c_str(),
                 config.list_flag.c_str(), executable, config.case_flag.c_str(), config.env_jobs.c_str(),
                 config.env_filter.c_str(), config.env_output.c_str());
}

inline bool parse_options(int argc, char ** argv, const Config & config, Options & options) {
    options.executable = argc > 0 ? argv[0] : config.suite_name.c_str();
    if (!config.env_filter.empty()) {
        if (const char * env_filter = std::getenv(config.env_filter.c_str())) {
            options.filter         = env_filter;
            options.runner_control = true;
        }
    }
    if (!config.env_jobs.empty()) {
        if (const char * env_jobs = std::getenv(config.env_jobs.c_str())) {
            if (!parse_positive_size(env_jobs, options.jobs)) {
                std::fprintf(stderr, "invalid %s value: %s\n", config.env_jobs.c_str(), env_jobs);
                return false;
            }
            options.runner_control = true;
        }
    }
    if (!config.env_output.empty()) {
        if (const char * env_output = std::getenv(config.env_output.c_str())) {
            if (!parse_output_mode(env_output, options.output)) {
                std::fprintf(stderr, "invalid %s value: %s\n", config.env_output.c_str(), env_output);
                return false;
            }
            options.runner_control = true;
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == config.list_flag) {
            options.list           = true;
            options.runner_control = true;
        } else if (arg == config.case_flag) {
            if (++i >= argc) {
                std::fprintf(stderr, "%s requires a case name\n", config.case_flag.c_str());
                return false;
            }
            options.single_case    = argv[i];
            options.runner_control = true;
        } else if (arg == config.filter_flag) {
            if (++i >= argc) {
                std::fprintf(stderr, "%s requires a regex\n", config.filter_flag.c_str());
                return false;
            }
            options.filter         = argv[i];
            options.runner_control = true;
        } else if (arg == config.jobs_flag) {
            if (++i >= argc || !parse_positive_size(argv[i], options.jobs)) {
                std::fprintf(stderr, "%s requires a positive integer\n", config.jobs_flag.c_str());
                return false;
            }
            options.runner_control = true;
        } else if (arg == config.output_flag) {
            if (++i >= argc || !parse_output_mode(argv[i], options.output)) {
                std::fprintf(stderr, "%s requires failed, all, or none\n", config.output_flag.c_str());
                return false;
            }
            options.runner_control = true;
        } else if (arg == "--help" || arg == "-h") {
            options.help           = true;
            options.runner_control = true;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (options.jobs == 0) {
        options.jobs = default_jobs(config);
    }
    return true;
}

inline const Case * find_case(const std::vector<Case> & cases, const std::string & name) {
    for (const Case & test_case : cases) {
        if (test_case.name == name) {
            return &test_case;
        }
    }
    return nullptr;
}

inline int run_single_case_in_process(const Config &           config,
                                      const std::vector<Case> & cases,
                                      const std::string &       name,
                                      bool                      has_device) {
    const Case * test_case = find_case(cases, name);
    if (test_case == nullptr) {
        std::fprintf(stderr, "unknown test case: %s\n", name.c_str());
        return 2;
    }
    if (test_case->requires_device && !has_device) {
        std::fprintf(stderr, "test skipped: no device available for %s\n", config.suite_name.c_str());
        return 0;
    }
    test_case->run();
    return 0;
}

inline Result run_single_case_child(const Config & config, const Options & options, const Case & test_case) {
    Result result;
    result.name = test_case.name;

    const auto start = std::chrono::steady_clock::now();

#if defined(_WIN32)
    result.output    = "process-isolated test runner is not supported on this platform\n";
    result.exit_code = 127;
    result.elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    return result;
#else
    int pipe_fds[2] = { -1, -1 };
    if (pipe(pipe_fds) != 0) {
        result.output    = "failed to create output pipe\n";
        result.exit_code = 127;
        return result;
    }

    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        close(pipe_fds[0]);
        dup2(pipe_fds[1], STDOUT_FILENO);
        dup2(pipe_fds[1], STDERR_FILENO);
        close(pipe_fds[1]);

        std::vector<char *> child_argv;
        child_argv.push_back(const_cast<char *>(options.executable.c_str()));
        child_argv.push_back(const_cast<char *>(config.case_flag.c_str()));
        child_argv.push_back(const_cast<char *>(test_case.name.c_str()));
        child_argv.push_back(nullptr);
        execvp(child_argv[0], child_argv.data());
        std::fprintf(stderr, "failed to exec %s: %s\n", child_argv[0], std::strerror(errno));
        _exit(127);
    }

    close(pipe_fds[1]);
    if (pid < 0) {
        close(pipe_fds[0]);
        result.output    = "failed to fork child test process\n";
        result.exit_code = 127;
        return result;
    }

    char buffer[4096];
    for (;;) {
        const ssize_t bytes = read(pipe_fds[0], buffer, sizeof(buffer));
        if (bytes > 0) {
            result.output.append(buffer, static_cast<size_t>(bytes));
            continue;
        }
        if (bytes == 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        result.output += "failed to read child output\n";
        break;
    }
    close(pipe_fds[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        result.output += "failed to wait for child test process\n";
        result.exit_code = 127;
        return result;
    }

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
        result.passed    = result.exit_code == 0;
    } else if (WIFSIGNALED(status)) {
        result.signal_code = WTERMSIG(status);
        result.passed      = false;
    }

    result.elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    return result;
#endif
}

inline bool should_print_output(OutputMode mode, const Result & result) {
    if (mode == OutputMode::All) {
        return true;
    }
    if (mode == OutputMode::None) {
        return false;
    }
    return !result.passed;
}

inline int run_cases_parallel(const Config &            config,
                              const Options &           options,
                              const std::vector<Case> & selected_cases) {
    if (selected_cases.empty()) {
        std::printf("No %s cases selected\n", config.suite_name.c_str());
        return 0;
    }

    const size_t        jobs = std::min(options.jobs, selected_cases.size());
    std::vector<Result> results(selected_cases.size());
    std::atomic<size_t> next_index(0);
    std::atomic<size_t> finished(0);
    std::mutex          print_mutex;
    const auto          suite_start = std::chrono::steady_clock::now();

    std::printf("Running %zu %s case(s) with %zu worker process(es)\n", selected_cases.size(),
                config.suite_name.c_str(), jobs);

    auto worker = [&] {
        for (;;) {
            const size_t index = next_index.fetch_add(1);
            if (index >= selected_cases.size()) {
                return;
            }
            Result       result = run_single_case_child(config, options, selected_cases[index]);
            const size_t done   = finished.fetch_add(1) + 1;
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::printf("%s %4zu/%zu %-86s %8lld ms\n", result.passed ? "[PASS]" : "[FAIL]", done,
                            selected_cases.size(), result.name.c_str(), static_cast<long long>(result.elapsed_ms));
                if (!result.passed) {
                    if (result.signal_code != 0) {
                        std::printf("       terminated by signal %d\n", result.signal_code);
                    } else {
                        std::printf("       exit code %d\n", result.exit_code);
                    }
                }
                if (should_print_output(options.output, result) && !result.output.empty()) {
                    std::printf("----- output: %s -----\n%s", result.name.c_str(), result.output.c_str());
                    if (result.output.back() != '\n') {
                        std::printf("\n");
                    }
                    std::printf("----- end output: %s -----\n", result.name.c_str());
                }
                std::fflush(stdout);
            }
            results[index] = std::move(result);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(jobs);
    for (size_t i = 0; i < jobs; ++i) {
        workers.emplace_back(worker);
    }
    for (std::thread & thread : workers) {
        thread.join();
    }

    size_t passed = 0;
    for (const Result & result : results) {
        if (result.passed) {
            ++passed;
        }
    }
    const size_t  failed = results.size() - passed;
    const int64_t elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - suite_start).count();

    std::printf("\n%s summary: %zu passed, %zu failed, %zu total, %lld ms\n", config.suite_name.c_str(), passed, failed,
                results.size(), static_cast<long long>(elapsed_ms));
    if (failed != 0) {
        std::printf("Failed cases:\n");
        for (const Result & result : results) {
            if (!result.passed) {
                std::printf("  %s\n", result.name.c_str());
            }
        }
    }
    return failed == 0 ? 0 : 1;
}

inline bool select_cases(const std::vector<Case> & cases,
                         const std::string &       filter_text,
                         bool                      has_device,
                         bool                      include_unavailable_device_cases,
                         Selection &               selection) {
    try {
        std::regex filter;
        const bool has_filter = !filter_text.empty();
        if (has_filter) {
            filter = std::regex(filter_text);
        }
        for (const Case & test_case : cases) {
            if (has_filter && !std::regex_search(test_case.name, filter)) {
                continue;
            }
            ++selection.matched;
            if (!has_device && test_case.requires_device && !include_unavailable_device_cases) {
                ++selection.skipped_device;
                continue;
            }
            selection.cases.push_back(test_case);
        }
    } catch (const std::regex_error & error) {
        std::fprintf(stderr, "invalid test filter regex: %s\n", error.what());
        return false;
    }
    return true;
}

inline void print_case_list(const std::vector<Case> & cases) {
    for (const Case & test_case : cases) {
        std::printf("[%s] %s\n", test_case.requires_device ? "device" : "host", test_case.name.c_str());
    }
}

inline bool validate_case_names(const std::vector<Case> & cases) {
    for (size_t i = 0; i < cases.size(); ++i) {
        for (size_t j = i + 1; j < cases.size(); ++j) {
            if (cases[i].name == cases[j].name) {
                std::fprintf(stderr, "duplicate test case: %s\n", cases[i].name.c_str());
                return false;
            }
        }
    }
    return true;
}

class Suite {
  public:
    explicit Suite(Config config) : config_(std::move(config)) {}

    template <typename F> void device_case(std::string name, F && run) {
        add_case(cases_, std::move(name), std::function<void()>(std::forward<F>(run)), true);
    }

    template <typename F> void host_case(std::string name, F && run) {
        add_case(cases_, std::move(name), std::function<void()>(std::forward<F>(run)), false);
    }

    bool parse_options(int argc, char ** argv, Options & options) const {
        return test_runner::parse_options(argc, argv, config_, options);
    }

    void print_usage(const char * executable) const { test_runner::print_usage(config_, executable); }

    int run(int argc, char ** argv, bool has_device) const {
        Options options;
        if (!parse_options(argc, argv, options)) {
            print_usage(argc > 0 ? argv[0] : config_.suite_name.c_str());
            return 2;
        }
        return run(options, has_device);
    }

    int run(const Options & options, bool has_device) const {
        if (options.help) {
            print_usage(options.executable.c_str());
            return 0;
        }
        if (!validate_case_names(cases_)) {
            return 2;
        }
        if (!options.single_case.empty()) {
            return run_single_case_in_process(config_, cases_, options.single_case, has_device);
        }

        Selection selection;
        if (!select_cases(cases_, options.filter, has_device, options.list, selection)) {
            return 2;
        }

        if (options.list) {
            print_case_list(selection.cases);
            return 0;
        }

        if (selection.skipped_device != 0) {
            std::fprintf(stderr, "test skipped: no device available for %s (%zu device case(s))\n",
                         config_.suite_name.c_str(), selection.skipped_device);
            if (selection.cases.empty()) {
                return 0;
            }
        }

        return run_cases_parallel(config_, options, selection.cases);
    }

  private:
    Config            config_;
    std::vector<Case> cases_;
};

}  // namespace test_runner
