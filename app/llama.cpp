#include "build-info.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// visible
int llama_server(int argc, char ** argv);
int llama_cli(int argc, char ** argv);

// hidden
int llama_completion(int argc, char ** argv);
int llama_bench(int argc, char ** argv);
int llama_batched_bench(int argc, char ** argv);
int llama_fit_params(int argc, char ** argv);
int llama_quantize(int argc, char ** argv);
int llama_perplexity(int argc, char ** argv);

static int help(int argc, char ** argv);
static int version(int argc, char ** argv);

struct command {
    const char * name;
    const char * desc;
    std::vector<std::string> aliases;
    bool hidden;
    int (*func)(int, char **);
};

static const command cmds[] = {
    {"serve",         "HTTP API server",                                    {"server"},   false, llama_server       },
    {"cli",           "Command-line interactive interface",                 {"client"},   false, llama_cli          },
    {"completion",    "Text completion",                                    {"complete"}, true,  llama_completion   },
    {"bench",         "Benchmark prompt processing and text generation",    {},           true,  llama_bench        },
    {"batched-bench", "Benchmark batched decoding performance",             {},           true,  llama_batched_bench},
    {"fit-params",    "Compute parameters to fit a model in device memory", {},           true,  llama_fit_params   },
    {"quantize",      "Quantize a model",                                   {},           true,  llama_quantize     },
    {"perplexity",    "Compute model perplexity and KL divergence",         {},           true,  llama_perplexity   },
    {"version",       "Show version",                                       {},           true,  version            },
    {"help",          "Show available commands",                            {},           true,  help               },
};

static int version(int argc, char ** argv) {
    printf("%s\n", llama_build_info());
    return 0;
}

static int help(int argc, char ** argv) {
    const bool show_all = argc >= 2 && std::string(argv[1]) == "all";

    printf("Usage: llama <command> [options]\n\nAvailable commands:\n");

    for (const auto & cmd : cmds) {
        if (show_all || !cmd.hidden) {
            printf("  %-15s %s\n", cmd.name, cmd.desc);
        }
    }
    printf("\nRun 'llama <command> --help' for command-specific usage.\n");

    return 0;
}

static bool matches(const std::string & arg, const command & cmd) {
    if (arg == cmd.name) {
        return true;
    }
    for (const auto & alias : cmd.aliases) {
        if (arg == alias) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    const std::string arg = argc >= 2 ? argv[1] : "help";

    for (const auto & cmd : cmds) {
        if (matches(arg, cmd)) {

            // router spawns children through this same binary, it needs the
            // subcommand to relaunch as 'llama serve' and not bare options
#ifdef _WIN32
            _putenv_s("LLAMA_APP_CMD", cmd.name);
#else
            setenv("LLAMA_APP_CMD", cmd.name, 1);
#endif
            return cmd.func(argc - 1, argv + 1);
        }
    }

    fprintf(stderr, "error: unknown command '%s'\n", arg.c_str());
    return 1;
}
