#pragma once

#include "common.h"
#include "download.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

// pseudo-env variable to identify preset-only arguments
#define COMMON_ARG_PRESET_LOAD_ON_STARTUP "__PRESET_LOAD_ON_STARTUP"
#define COMMON_ARG_PRESET_STOP_TIMEOUT    "__PRESET_STOP_TIMEOUT"

//
// CLI argument parsing
//

struct common_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::set<enum llama_example> excludes = {};
    // Tessera fork: when non-empty, this flag is only visible to the
    // listed subcommands. Empty set = visible regardless of subcommand
    // (top-level / "any"). Only meaningful when LLAMA_EXAMPLE_TESSERA
    // is in `examples`; ignored otherwise.
    std::set<enum tessera_subcommand> tessera_sc = {};
    std::vector<const char *> args;
    std::vector<const char *> args_neg;  // for negated args like --no-xxx
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    bool is_sampling = false; // is current arg a sampling param?
    bool is_spec = false; // is current arg a speculative decoding param?
    bool is_preset_only = false; // is current arg preset-only (not treated as CLI arg)
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (common_params & params, int) = nullptr;
    void (*handler_bool)   (common_params & params, bool) = nullptr;

    common_arg() = default;

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, int)
    ) : args(args), value_hint(value_hint), help(help), handler_int(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(common_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::initializer_list<const char *> & args_neg,
        const std::string & help,
        void (*handler)(common_params & params, bool)
    ) : args(args), args_neg(args_neg), help(help), handler_bool(handler) {}

    // support 2 values for arg
    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const char * value_hint_2,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &, const std::string &)
    ) : args(args), value_hint(value_hint), value_hint_2(value_hint_2), help(help), handler_str_str(handler) {}

    common_arg & set_examples(std::initializer_list<enum llama_example> examples);
    common_arg & set_excludes(std::initializer_list<enum llama_example> excludes);
    // Tessera fork: tag a flag with the subcommand(s) it belongs to. A
    // flag with no subcommand tag is visible regardless of the active
    // subcommand. Multiple subcommand tags mean the flag is visible
    // under any of them.
    common_arg & set_tessera_sc(std::initializer_list<enum tessera_subcommand> sc);
    common_arg & set_env(const char * env);
    common_arg & set_sampling();
    common_arg & set_spec();
    common_arg & set_preset_only();
    bool in_example(enum llama_example ex);
    bool is_exclude(enum llama_example ex);
    bool get_value_from_env(std::string & output) const;
    bool has_value_from_env() const;
    std::string to_string() const;

    // for using as key in std::map
    bool operator<(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) < 0;
    }
    bool operator==(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) == 0;
    }

    // get all args and env vars (including negated args/env)
    std::vector<std::string> get_args() const;
    std::vector<std::string> get_env() const;
};

namespace common_arg_utils {
    bool is_truthy(const std::string & value);
    bool is_falsey(const std::string & value);
    bool is_autoy(const std::string & value);
}

struct common_params_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    // Tessera fork: the active subcommand. TESSERA_SC_NONE means the
    // main quantize path (no subcommand given). When ex ==
    // LLAMA_EXAMPLE_TESSERA, this controls per-subcommand flag
    // visibility via common_arg::tessera_sc.
    enum tessera_subcommand tessera_sc = TESSERA_SC_NONE;
    common_params & params;
    std::vector<common_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    common_params_context(common_params & params) : params(params) {}
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// parse input arguments from CLI into a map
bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map);

// populate preset-only arguments
// these arguments are not treated as command line arguments
// see: https://github.com/ggml-org/llama.cpp/issues/18163
void common_params_add_preset_options(std::vector<common_arg> & args);

// Dispatch a single Tessera-flavored arg (--tessera-*, --calib-*,
// --progress-file, --quantize-db) at argv[i] to the handler registered
// in common_params_parse_ex via the add_opt path. Intended for tools
// (llama-quantize) that hand-roll their arg loop instead of calling
// common_params_parse.
// Returns argv slots consumed (1 = switch, 2 = valued), 0 if argv[i] is
// not a Tessera flag, or -1 on a validation error (message written to err).
int common_arg_dispatch_tessera(int argc, char ** argv, int i, std::string & err);

struct common_models_handler {
    common_download_hf_plan plan;
    common_download_hf_plan plan_spec;
    common_download_hf_plan plan_voc;
    common_download_opts opts;
};

// initialize downloading opts and hf_plan if needed, but does not download anything yet
common_models_handler common_models_handler_init(const common_params & params, llama_example curr_ex);

// check if the model is a preset repo (i.e. has a preset file)
bool common_models_handler_is_preset_repo(const common_models_handler & handler);

// download and update params with the downloaded model path
void common_models_handler_apply(common_models_handler & handler, common_params & params, common_download_callback * callback = nullptr);

// initialize argument parser context - used by test-arg-parser and preset
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);
