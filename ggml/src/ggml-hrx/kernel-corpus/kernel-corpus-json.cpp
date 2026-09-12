#include "kernel-corpus-json.h"

#include <nlohmann/json.hpp>

namespace ggml::hrx {
namespace {

const char * kernel_resource_access_name(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read:
            return "read";
        case ResourceAccess::Write:
            return "write";
        case ResourceAccess::ReadWrite:
            return "read_write";
    }
    return "unknown";
}

nlohmann::ordered_json string_span_json(KernelSpan<const char *> values) {
    nlohmann::ordered_json result = nlohmann::ordered_json::array();
    for (const char * value : values) {
        result.push_back(value != nullptr ? value : "");
    }
    return result;
}

nlohmann::ordered_json source_ref_span_json(KernelSpan<KernelSourceRef> values) {
    nlohmann::ordered_json result = nlohmann::ordered_json::array();
    for (const KernelSourceRef & value : values) {
        result.push_back(value.path != nullptr ? value.path : "");
    }
    return result;
}

nlohmann::ordered_json compile_config_json(KernelSpan<KernelCompileConfig> values) {
    nlohmann::ordered_json result = nlohmann::ordered_json::object();
    for (const KernelCompileConfig & value : values) {
        result[value.key != nullptr ? value.key : ""] = value.value != nullptr ? value.value : "";
    }
    return result;
}

}  // namespace

std::string serialize_kernel_corpus_json(const KernelCorpus & corpus) {
    nlohmann::ordered_json root = {
        { "schema",            corpus.schema                   },
        { "upstream_revision", corpus.upstream_revision        },
        { "corpus_digest",     corpus.corpus_digest            },
        { "recipe_digest",     corpus.recipe_digest            },
        { "plan_case_count",   corpus.plan_case_count          },
        { "kernels",           nlohmann::ordered_json::array() },
    };
    for (const KernelDefinition & kernel : corpus.kernels) {
        nlohmann::ordered_json item = {
            { "family",              kernel.family                              },
            { "name",                kernel.name                                },
            { "id",                  kernel.id                                  },
            { "source",              kernel.source                              },
            { "dependencies",        string_span_json(kernel.dependencies)      },
            { "symbol",              kernel.symbol                              },
            { "backend",             kernel.backend                             },
            { "target_selector",     kernel.target_selector                     },
            { "compile_config",      compile_config_json(kernel.compile_config) },
            { "scalar_parameters",   string_span_json(kernel.scalar_parameters) },
            { "source_digest",       kernel.source_digest                       },
            { "compile_recipe",
             {
                  { "mode", kernel.compile_recipe.mode },
                  { "link_module", kernel.compile_recipe.link_module },
                  { "primary_sources", source_ref_span_json(kernel.compile_recipe.primary_sources) },
                  { "library_sources", source_ref_span_json(kernel.compile_recipe.library_sources) },
              }                                                                 },
            { "workload_parameters", nlohmann::ordered_json::array()            },
            { "launch_parameters",   nlohmann::ordered_json::array()            },
            { "bindings",            nlohmann::ordered_json::array()            },
        };
        for (const KernelScalarDefinition & parameter : kernel.workload_parameters) {
            item["workload_parameters"].push_back({
                { "name", parameter.name },
                { "type", parameter.type }
            });
        }
        for (const KernelScalarDefinition & parameter : kernel.launch_parameters) {
            item["launch_parameters"].push_back({
                { "name", parameter.name },
                { "type", parameter.type }
            });
        }
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            item["bindings"].push_back({
                { "name",   binding.name                                },
                { "access", kernel_resource_access_name(binding.access) }
            });
        }
        root["kernels"].push_back(std::move(item));
    }
    return root.dump();
}

}  // namespace ggml::hrx
