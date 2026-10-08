#pragma once

#include "kernel-corpus-catalog.h"
#include "kernel-types.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace ggml::hrx {

template <typename T> struct KernelSpan {
    const T * items = nullptr;
    size_t    count = 0;

    const T * begin() const { return items; }

    const T * end() const { return items == nullptr ? nullptr : items + count; }

    const T * data() const { return items; }

    size_t size() const { return count; }

    bool empty() const { return count == 0; }

    const T & operator[](size_t index) const { return items[index]; }

    const T & front() const { return items[0]; }
};

struct KernelCompileConfig {
    const char * key   = "";
    const char * value = "";
};

struct KernelBindingDefinition {
    const char *   name   = "";
    ResourceAccess access = ResourceAccess::Read;
};

struct KernelScalarDefinition {
    const char * name = "";
    const char * type = "";
};

enum KernelSourceFormat {
    KERNEL_SOURCE_FORMAT_TEXT,
    KERNEL_SOURCE_FORMAT_BINARY,
};

struct KernelSourceSpan {
    const char *       data;
    size_t             length;
    KernelSourceFormat format;
};

struct KernelSource {
    KernelSourceSpan         source;
    const KernelSourceSpan * dependencies;
    size_t                   dependency_count;
};

struct KernelSourceRef {
    const char *         path     = "";
    const KernelSource * contents = nullptr;
};

struct KernelCompileRecipe {
    const char *                mode        = "";
    const char *                link_module = "";
    KernelSpan<KernelSourceRef> primary_sources;
    KernelSpan<KernelSourceRef> library_sources;
};

struct KernelDefinition {
    const char *                        family = "";
    const char *                        name   = "";
    uint64_t                            id     = kUncatalogedKernelId;
    const char *                        source = "";
    KernelSpan<const char *>            dependencies;
    const char *                        symbol          = "";
    const char *                        backend         = "";
    const char *                        target_selector = "";
    KernelSpan<KernelCompileConfig>     compile_config;
    KernelSpan<const char *>            scalar_parameters;
    KernelSpan<KernelBindingDefinition> bindings;
    const char *                        source_digest = "";
    KernelSpan<KernelScalarDefinition>  workload_parameters;
    KernelSpan<KernelScalarDefinition>  launch_parameters;
    KernelCompileRecipe                 compile_recipe;
};

struct KernelCorpus {
    const char *                 schema            = "ggml-hrx-kernel-corpus-v2";
    const char *                 upstream_revision = "";
    const char *                 corpus_digest     = "";
    const char *                 recipe_digest     = "";
    size_t                       plan_case_count   = 0;
    KernelSpan<KernelDefinition> kernels;
};

enum class KernelResolveStatus : uint8_t {
    Found,
    UncatalogedKernel,
    MissingActiveCorpusEntry,
    HashCollision,
    UnsupportedTarget,
};

struct KernelResolveResult {
    KernelResolveStatus      status     = KernelResolveStatus::MissingActiveCorpusEntry;
    const KernelDefinition * definition = nullptr;

    bool found() const { return status == KernelResolveStatus::Found && definition != nullptr; }
};

const KernelSource * get_kernel_source(const char * source_path);
const KernelCorpus & get_qwen_kernel_corpus();
KernelResolveResult  resolve_kernel_definition(const KernelCorpus & corpus,
                                               const std::string &  target,
                                               uint64_t             kernel_id);
const char *         kernel_resolve_status_name(KernelResolveStatus status);
std::string          kernel_definition_name(const KernelDefinition & definition);
std::string          kernel_definition_name_or_id(const KernelDefinition * definition, uint64_t kernel_id);
std::string          format_kernel_resolve_error(const KernelResolveResult & result, uint64_t kernel_id);
VerificationResult   verify_kernel_corpus(const KernelCorpus & corpus);
std::string          format_kernel_corpus(const KernelCorpus & corpus);

}  // namespace ggml::hrx
