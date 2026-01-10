#pragma once

#include "common.h"
#include "arg.h"

#include <string>
#include <vector>
#include <map>
#include <set>

//
// INI preset parser and writer
//

constexpr const char * COMMON_PRESET_DEFAULT_NAME = "default";

struct common_preset_context;

struct common_preset {
    std::string name;

    // options are stored as common_arg to string mapping, representing CLI arg and its value
    std::map<common_arg, std::string> options;

    // convert preset to CLI argument list
    std::vector<std::string> to_args(const std::string & bin_path = "") const;

    // convert preset to INI format string
    std::string to_ini() const;

    // TODO: maybe implement to_env() if needed

    // modify preset options where argument is identified by its env variable
    void set_option(const common_preset_context & ctx, const std::string & env, const std::string & value);

    // unset option by its env variable
    void unset_option(const std::string & env);

    // get option value by its env variable, return false if not found
    bool get_option(const std::string & env, std::string & value) const;

    // merge another preset into this one, overwriting existing options
    void merge(const common_preset & other);

    // apply preset options to common_params
    void apply_to_params(common_params & params) const;
};

// interface for multiple presets in one file
using common_presets = std::map<std::string, common_preset>;

// context for loading and editing presets
struct common_preset_context {
    common_params default_params; // unused for now
    common_params_context ctx_params;
    std::map<std::string, common_arg> key_to_opt;

    bool filter_allowed_keys = false;
    std::set<std::string> allowed_keys;

    // if only_remote_allowed is true, only accept whitelisted keys
    common_preset_context(llama_example ex, bool only_remote_allowed = false);

    // load presets from INI file
    common_presets load_from_ini(const std::string & path, common_preset & global) const;

    // generate presets from cached models
    common_presets load_from_cache() const;

    // generate presets from local models directory
    // for the directory structure, see "Using multiple models" in server/README.md
    common_presets load_from_models_dir(const std::string & models_dir) const;

    // generate one preset from CLI arguments
    common_preset load_from_args(int argc, char ** argv) const;

    // cascade multiple presets if exist on both: base < added
    // if preset does not exist in base, it will be added without modification
    common_presets cascade(const common_presets & base, const common_presets & added) const;

    // apply presets over a base preset (same idea as CSS cascading)
    common_presets cascade(const common_preset & base, const common_presets & presets) const;
};
