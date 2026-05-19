#include <cstdio>
#include <string>
#include <vector>

int llama_server(int argc, char ** argv);
int llama_cli(int argc, char ** argv);

// hidden
int llama_completion(int argc, char ** argv);
int llama_bench(int argc, char ** argv);
static int help(int argc, char ** argv);

struct command {
    const char * name;
    const char * desc;
    std::vector<std::string> aliases;
    bool hidden;
    int (*func)(int, char **);
};

static const command cmds[] = {
    {"serve",      "HTTP API server",                    {"server"},   false, llama_server     },
    {"cli",        "Command-line interactive interface", {"client"},   false, llama_cli        },
    {"completion", "Text completion",                    {"complete"}, true,  llama_completion },
    {"bench",      "Benchmarking tool",                  {},           true,  llama_bench      },
    {"help",       "Show available commands",            {},           true,  help             },
};

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
            return cmd.func(argc - 1, argv + 1);
        }
    }

    fprintf(stderr, "error: unknown command '%s'\n", arg.c_str());
    return 1;
}
