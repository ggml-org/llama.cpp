#include "arg.h"
#include "common.h"

#include <fstream>
#include <string>

// Export usage message (-h) to markdown format

static void write_table_header(std::ofstream & file) {
    file << "| Argument | Explanation |\n";
    file << "| -------- | ----------- |\n";
}

static void write_table_entry(std::ofstream & file, const common_arg & opt) {
    file << "| `";
    // args
    auto all_args = opt.get_args();
    for (const auto & arg : all_args) {
    if (arg == all_args.front()) {
            file << arg;
            if (all_args.size() > 1) file << ", ";
        } else {
            file << arg << (arg != all_args.back() ? ", " : "");
        }
    }
    // value hint
    if (opt.value_hint) {
        std::string md_value_hint(opt.value_hint);
        string_replace_all(md_value_hint, "|", "\\|");
        file << " " << md_value_hint;
    }
    if (opt.value_hint_2) {
        std::string md_value_hint_2(opt.value_hint_2);
        string_replace_all(md_value_hint_2, "|", "\\|");
        file << " " << md_value_hint_2;
    }
    // help text
    std::string md_help(opt.help);
    string_replace_all(md_help, "\n", "<br/>");
    string_replace_all(md_help, "|", "\\|");
    file << "` | " << md_help << " |\n";
}

static void write_table(std::ofstream & file, std::vector<common_arg *> & opts) {
    write_table_header(file);
    for (const auto & opt : opts) {
        write_table_entry(file, *opt);
    }
}

static void export_md(std::string fname, llama_example ex) {
    std::ofstream file(fname, std::ofstream::out | std::ofstream::trunc);

    common_params params;
    auto ctx_arg = common_params_parser_init(params, ex);

    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sparam_options;
    std::vector<common_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sparam) {
            sparam_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }

    file << "**Common params**\n\n";
    write_table(file, common_options);
    file << "\n\n**Sampling params**\n\n";
    write_table(file, sparam_options);
    file << "\n\n**Example-specific params**\n\n";
    write_table(file, specific_options);
}

int main(int, char **) {
    export_md("autogen-main.md", LLAMA_EXAMPLE_COMPLETION);
    export_md("autogen-server.md", LLAMA_EXAMPLE_SERVER);

    return 0;
}
