#include "dispatch/command-program-diagnostics.h"
#include "dispatch/command-program.h"
#include "dispatch/dispatch-scheduler.h"
#include "graph/graph-diagnostics.h"
#include "kernel-corpus/kernel-corpus.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path graph;
    std::filesystem::path output_directory;
    std::string           target;
};

static void print_usage(const char * program) {
    std::cerr << "usage: " << program << " --graph <graph.json> [--target <gfx1100|gfx1151>] [--out <dir>]\n";
}

static bool parse_options(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "--graph" && i + 1 < argc) {
            options.graph = argv[++i];
            continue;
        }
        if (arg == "--target" && i + 1 < argc) {
            options.target = argv[++i];
            continue;
        }
        if (arg == "--out" && i + 1 < argc) {
            options.output_directory = argv[++i];
            continue;
        }
        std::cerr << "unknown or incomplete argument: " << arg << '\n';
        print_usage(argv[0]);
        return false;
    }
    if (options.graph.empty()) {
        std::cerr << "--graph is required\n";
        print_usage(argv[0]);
        return false;
    }
    return true;
}

static std::string read_file(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

static void write_file(const std::filesystem::path & path, const std::string & contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create " + path.string());
    }
    output << contents;
    if (contents.empty() || contents.back() != '\n') {
        output << '\n';
    }
}

static void write_output(const Options & options, const std::string & name, const std::string & contents) {
    if (options.output_directory.empty()) {
        std::cout << "== " << name << " ==\n" << contents;
        if (contents.empty() || contents.back() != '\n') {
            std::cout << '\n';
        }
        return;
    }
    write_file(options.output_directory / name, contents);
}

}  // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        return 1;
    }

    const std::string contents = read_file(options.graph);
    if (contents.empty()) {
        std::cerr << "cannot read graph snapshot: " << options.graph << '\n';
        return 1;
    }

    ggml::hrx::GraphSnapshotLoadResult snapshot = ggml::hrx::load_graph_snapshot_json(contents);
    if (!snapshot.valid()) {
        for (const std::string & error : snapshot.status.errors()) {
            std::cerr << error << '\n';
        }
        return 1;
    }

    if (options.target.empty()) {
        options.target = snapshot.target;
    }
    if (options.target.empty()) {
        std::cerr << "target is missing from both --target and graph snapshot\n";
        return 1;
    }

    write_output(options, "graph.txt",
                 ggml::hrx::format_graph_snapshot_text(snapshot.graph, options.target, snapshot.uid));
    write_output(options, "graph.json",
                 ggml::hrx::serialize_graph_snapshot_json(snapshot.graph, options.target, snapshot.uid));

    ggml::hrx::DispatchScheduler           scheduler;
    ggml::hrx::DispatchScheduleDiagnostics diagnostics;
    const bool scheduled = scheduler.schedule_graph(snapshot.graph, { options.target }, &diagnostics);
    write_output(options, "schedule.txt",
                 ggml::hrx::format_schedule_diagnostics_text(snapshot.graph, scheduler.plan(), diagnostics));
    write_output(options, "schedule.json",
                 ggml::hrx::serialize_schedule_diagnostics_json(snapshot.graph, scheduler.plan(), diagnostics));
    if (!scheduled) {
        write_output(options, "unmatched.txt",
                     ggml::hrx::format_schedule_diagnostics_text(snapshot.graph, scheduler.plan(), diagnostics));
        write_output(options, "unmatched.json",
                     ggml::hrx::serialize_schedule_diagnostics_json(snapshot.graph, scheduler.plan(), diagnostics));
        return 2;
    }

    const ggml::hrx::KernelCorpus & corpus = ggml::hrx::get_qwen_kernel_corpus();
    ggml::hrx::CommandProgram       commands =
        ggml::hrx::build_command_program(snapshot.graph, scheduler.plan(), corpus, options.target);
    write_output(options, "commands.txt", ggml::hrx::format_command_program(commands));
    if (!commands.valid()) {
        for (const std::string & error : commands.status.errors()) {
            std::cerr << error << '\n';
        }
        return 3;
    }

    const ggml::hrx::VerificationResult verification =
        ggml::hrx::verify_command_program(commands, corpus, options.target);
    if (!verification.valid()) {
        for (const std::string & error : verification.status.errors()) {
            std::cerr << error << '\n';
        }
        return 4;
    }

    return 0;
}
