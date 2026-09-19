#include "kernel-corpus.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>

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

struct KernelSourceRecordEntry {
    const char *         source_path;
    const KernelSource * source;
};

static bool string_equal(const char * lhs, const char * rhs) {
    return std::strcmp(lhs != nullptr ? lhs : "", rhs != nullptr ? rhs : "") == 0;
}

static bool string_empty(const char * value) {
    return value == nullptr || value[0] == 0;
}

static bool contains_source_ref(KernelSpan<KernelSourceRef> values, const char * path) {
    return std::find_if(values.begin(), values.end(),
                        [&](const KernelSourceRef & item) { return string_equal(item.path, path); }) != values.end();
}

static bool string_span_equal(KernelSpan<const char *> lhs, KernelSpan<const char *> rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                                                  [](const char * a, const char * b) { return string_equal(a, b); });
}

static bool scalar_span_equal(KernelSpan<KernelScalarDefinition> lhs, KernelSpan<KernelScalarDefinition> rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](const KernelScalarDefinition & a, const KernelScalarDefinition & b) {
                          return string_equal(a.name, b.name) && string_equal(a.type, b.type);
                      });
}

static bool binding_span_equal(KernelSpan<KernelBindingDefinition> lhs, KernelSpan<KernelBindingDefinition> rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](const KernelBindingDefinition & a, const KernelBindingDefinition & b) {
                          return string_equal(a.name, b.name) && a.access == b.access;
                      });
}

static bool kernel_variant_contract_equal(const KernelDefinition & lhs, const KernelDefinition & rhs) {
    return string_equal(lhs.backend, rhs.backend) && string_span_equal(lhs.scalar_parameters, rhs.scalar_parameters) &&
           scalar_span_equal(lhs.workload_parameters, rhs.workload_parameters) &&
           scalar_span_equal(lhs.launch_parameters, rhs.launch_parameters) &&
           binding_span_equal(lhs.bindings, rhs.bindings);
}

// clang-format off
#include "kernel-corpus-sources.inc"
#include "kernel-corpus-qwen.inc"
// clang-format on

}  // namespace

const KernelSource * get_kernel_source(const char * source_path) {
    if (source_path == nullptr) {
        return nullptr;
    }
    for (const KernelSourceRecordEntry & entry : kKernelSourceRecords) {
        if (std::strcmp(source_path, entry.source_path) == 0) {
            return entry.source;
        }
    }
    return nullptr;
}

const KernelCorpus & get_qwen_kernel_corpus() {
    return kQwenKernelCorpus;
}

KernelResolveResult resolve_kernel_definition(const KernelCorpus & corpus,
                                              const std::string &  target,
                                              uint64_t             kernel_id) {
    if (kernel_id == kUncatalogedKernelId) {
        return { KernelResolveStatus::UncatalogedKernel, nullptr };
    }
    const KernelDefinition * first_match     = nullptr;
    const KernelDefinition * default_variant = nullptr;
    bool                     target_mismatch = false;
    for (const KernelDefinition & kernel : corpus.kernels) {
        if (kernel.id != kernel_id) {
            continue;
        }
        if (first_match == nullptr) {
            first_match = &kernel;
        } else if (!string_equal(first_match->family, kernel.family) || !string_equal(first_match->name, kernel.name)) {
            return { KernelResolveStatus::HashCollision, nullptr };
        }
        if (string_equal(kernel.target_selector, target.c_str())) {
            return { KernelResolveStatus::Found, &kernel };
        }
        if (string_empty(kernel.target_selector)) {
            default_variant = &kernel;
        } else {
            target_mismatch = true;
        }
    }
    if (default_variant != nullptr) {
        return { KernelResolveStatus::Found, default_variant };
    }
    if (target_mismatch) {
        return { KernelResolveStatus::UnsupportedTarget, first_match };
    }
    return { KernelResolveStatus::MissingActiveCorpusEntry, nullptr };
}

const char * kernel_resolve_status_name(KernelResolveStatus status) {
    switch (status) {
        case KernelResolveStatus::Found:
            return "found";
        case KernelResolveStatus::UncatalogedKernel:
            return "uncataloged_kernel";
        case KernelResolveStatus::MissingActiveCorpusEntry:
            return "missing_active_corpus_entry";
        case KernelResolveStatus::HashCollision:
            return "hash_collision";
        case KernelResolveStatus::UnsupportedTarget:
            return "unsupported_target";
    }
    return "unknown";
}

std::string kernel_definition_name(const KernelDefinition & definition) {
    return std::string(definition.family != nullptr ? definition.family : "") + ":" +
           (definition.name != nullptr ? definition.name : "");
}

std::string kernel_definition_name_or_id(const KernelDefinition * definition, uint64_t kernel_id) {
    if (definition != nullptr) {
        return kernel_definition_name(*definition);
    }
    return "kernel_id=" + std::to_string(kernel_id);
}

std::string format_kernel_resolve_error(const KernelResolveResult & result, uint64_t kernel_id) {
    const std::string label = kernel_definition_name_or_id(result.definition, kernel_id);
    switch (result.status) {
        case KernelResolveStatus::Found:
            return "";
        case KernelResolveStatus::UncatalogedKernel:
            return "uncataloged kernel " + label;
        case KernelResolveStatus::MissingActiveCorpusEntry:
            return "cataloged kernel " + label + " is not available in the active corpus";
        case KernelResolveStatus::HashCollision:
            return "kernel catalog id collision while resolving " + label;
        case KernelResolveStatus::UnsupportedTarget:
            return "cataloged kernel " + label + " has no implementation for the requested target";
    }
    return "unknown kernel resolution failure for " + label;
}

VerificationResult verify_kernel_corpus(const KernelCorpus & corpus) {
    VerificationResult result;
    if (!string_equal(corpus.schema, "ggml-hrx-kernel-corpus-v2")) {
        result.status.log("unsupported kernel corpus schema");
    }
    if (string_empty(corpus.upstream_revision)) {
        result.status.log("kernel corpus has no upstream revision");
    }
    if (string_empty(corpus.corpus_digest)) {
        result.status.log("kernel corpus has no digest");
    }
    if (string_empty(corpus.recipe_digest)) {
        result.status.log("kernel corpus has no BUILD.bazel recipe digest");
    }
    if (corpus.plan_case_count == 0) {
        result.status.log("kernel corpus has no compile plan cases");
    }
    std::set<std::string>                           variants;
    std::map<std::string, const KernelDefinition *> contracts;
    for (const KernelDefinition & kernel : corpus.kernels) {
        if (string_empty(kernel.family) || string_empty(kernel.name) || string_empty(kernel.source) ||
            string_empty(kernel.symbol) || string_empty(kernel.backend) || string_empty(kernel.source_digest)) {
            result.status.log("kernel definition is incomplete");
        }
        if (kernel.id != kernel_catalog_id(kernel.family != nullptr ? kernel.family : "",
                                           kernel.name != nullptr ? kernel.name : "")) {
            result.status.log("kernel %s has an invalid catalog id", kernel.name != nullptr ? kernel.name : "");
        }
        const bool source_is_primary = contains_source_ref(kernel.compile_recipe.primary_sources, kernel.source);
        const bool source_is_library = contains_source_ref(kernel.compile_recipe.library_sources, kernel.source);
        if ((!string_equal(kernel.compile_recipe.mode, "direct") &&
             !string_equal(kernel.compile_recipe.mode, "archive")) ||
            kernel.compile_recipe.primary_sources.empty() || (!source_is_primary && !source_is_library) ||
            (string_equal(kernel.compile_recipe.mode, "archive") && string_empty(kernel.compile_recipe.link_module))) {
            result.status.log("kernel %s has an invalid BUILD compile recipe",
                              kernel.name != nullptr ? kernel.name : "");
        }
        for (const KernelSourceRef & source : kernel.compile_recipe.primary_sources) {
            if (string_empty(source.path) || source.contents == nullptr) {
                result.status.log("kernel %s has an invalid embedded primary source reference",
                                  kernel.name != nullptr ? kernel.name : "");
            }
        }
        for (const KernelSourceRef & source : kernel.compile_recipe.library_sources) {
            if (string_empty(source.path) || source.contents == nullptr) {
                result.status.log("kernel %s has an invalid embedded library source reference",
                                  kernel.name != nullptr ? kernel.name : "");
            }
        }
        const std::string full_name = std::string(kernel.family != nullptr ? kernel.family : "") + ":" +
                                      std::string(kernel.name != nullptr ? kernel.name : "");
        const std::string target_selector = kernel.target_selector != nullptr ? kernel.target_selector : "";
        if (!variants.insert(full_name + "@" + target_selector).second) {
            result.status.log("kernel corpus repeats target variant %s@%s", full_name.c_str(),
                              target_selector.empty() ? "default" : target_selector.c_str());
        }
        const auto contract = contracts.emplace(full_name, &kernel);
        if (!contract.second && !kernel_variant_contract_equal(*contract.first->second, kernel)) {
            result.status.log("kernel target variants disagree on ABI for %s", full_name.c_str());
        }
        std::set<std::string> binding_names;
        for (const KernelBindingDefinition & binding : kernel.bindings) {
            if (string_empty(binding.name) ||
                !binding_names.insert(binding.name != nullptr ? binding.name : "").second) {
                result.status.log("kernel %s has invalid binding names", kernel.name != nullptr ? kernel.name : "");
            }
        }
        if (kernel.bindings.size() == 0) {
            result.status.log("kernel %s has no binding ABI", kernel.name != nullptr ? kernel.name : "");
        }
    }
    return result;
}

std::string format_kernel_corpus(const KernelCorpus & corpus) {
    std::ostringstream out;
    out << "kernel-corpus " << corpus.schema << " revision=" << corpus.upstream_revision
        << " digest=" << corpus.corpus_digest << " recipe=" << corpus.recipe_digest
        << " kernels=" << corpus.kernels.size() << " plan_cases=" << corpus.plan_case_count << '\n';
    for (const KernelDefinition & kernel : corpus.kernels) {
        out << "  kernel " << kernel.family << ':' << kernel.name << " id=0x" << std::hex << kernel.id << std::dec
            << " backend=" << kernel.backend
            << " target=" << (string_empty(kernel.target_selector) ? "default" : kernel.target_selector) << " symbol=@"
            << kernel.symbol << " source=" << kernel.source << " sha256=" << kernel.source_digest << '\n';
        out << "    recipe " << kernel.compile_recipe.mode;
        if (!string_empty(kernel.compile_recipe.link_module)) {
            out << " module=" << kernel.compile_recipe.link_module;
        }
        out << " primary=";
        for (size_t i = 0; i < kernel.compile_recipe.primary_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.primary_sources[i].path;
        }
        out << " libraries=";
        for (size_t i = 0; i < kernel.compile_recipe.library_sources.size(); ++i) {
            out << (i ? "," : "") << kernel.compile_recipe.library_sources[i].path;
        }
        out << '\n';
        for (size_t i = 0; i < kernel.bindings.size(); ++i) {
            out << "    binding[" << i << "] " << kernel.bindings[i].name << ' '
                << kernel_resource_access_name(kernel.bindings[i].access) << '\n';
        }
        for (const KernelScalarDefinition & parameter : kernel.workload_parameters) {
            out << "    workload " << parameter.name << ' ' << parameter.type << '\n';
        }
        for (const KernelScalarDefinition & parameter : kernel.launch_parameters) {
            out << "    launch " << parameter.name << ' ' << parameter.type << '\n';
        }
    }
    return out.str();
}

}  // namespace ggml::hrx
