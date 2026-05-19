#include <string>
#include <vector>
#include <cstdio>

int llama_server(int argc, char ** argv);
int llama_completion(int argc, char ** argv);
int llama_cli(int argc, char ** argv);
int llama_bench(int argc, char ** argv);

struct command {
    std::string name;
    std::string desc;
    std::vector<const char *> aliases;
    int (*func)(int, char **);
};

static struct command cmds[] = {
    {"serve",     "HTTP API server",                     {"server"},   llama_server    },
    {"cli",        "Command-line interactive interface", {"client"},   llama_cli       },
    {"completion", "Text completion",                    {"complete"}, llama_completion},
    {"bench",      "Benchmarking tool",                  {},           llama_bench     },
};

static void print_usage(const char * prog) {
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Available commands:\n");

    for (const auto & cmd : cmds) {
        printf("  %-15s %s\n", cmd.name.data(), cmd.desc.data());
    }
    printf("\nRun '%s <command> --help' for command-specific usage.\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string arg = argv[1];

    for (const auto & cmd : cmds) {
        if (arg == cmd.name) {
            return cmd.func(argc - 1, argv + 1);
        }
        for (const auto & alias : cmd.aliases) {
            if (arg == alias) {
                return cmd.func(argc - 1, argv + 1);
            }
        }
    }

    fprintf(stderr, "error: unknown command '%s'\n\n", argv[1]);
    return 1;
}
