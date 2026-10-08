#include "command-program-dump.h"

#include "command-program-diagnostics.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ggml::hrx {
namespace {

static constexpr const char * kDumpDirectoryEnvironment = "GGML_HRX_DUMP_COMMAND_PROGRAM_DIR";

static void mix_hash(uint64_t & hash, uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

static uint64_t hash_string(const std::string & text) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const char c : text) {
        mix_hash(hash, static_cast<unsigned char>(c));
    }
    return hash;
}

static std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

static std::string dot_escape(const std::string & text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char c : text) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        if (c == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}

static std::string json_escape(const std::string & text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const unsigned char c : text) {
        switch (c) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\b':
                escaped += "\\b";
                break;
            case '\f':
                escaped += "\\f";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                if (c < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped.push_back(kHex[c >> 4]);
                    escaped.push_back(kHex[c & 0xf]);
                } else {
                    escaped.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return escaped;
}

static std::string kernel_name(const KernelCorpus & corpus, const std::string & target, uint64_t kernel_id) {
    const KernelResolveResult resolved = resolve_kernel_definition(corpus, target, kernel_id);
    if (resolved.found()) {
        return kernel_definition_name(*resolved.definition);
    }
    std::ostringstream out;
    out << "kernel_id=" << kernel_id;
    return out.str();
}

static void append_json_integer_map(std::ostringstream & out, const std::map<std::string, int64_t> & values) {
    out << '{';
    size_t index = 0;
    for (const auto & value : values) {
        if (index++ > 0) {
            out << ", ";
        }
        out << '"' << json_escape(value.first) << "\": " << value.second;
    }
    out << '}';
}

static void append_json_string_map(std::ostringstream & out, const std::map<std::string, std::string> & values) {
    out << '{';
    size_t index = 0;
    for (const auto & value : values) {
        if (index++ > 0) {
            out << ", ";
        }
        out << '"' << json_escape(value.first) << "\": \"" << json_escape(value.second) << '"';
    }
    out << '}';
}

static void append_kernel_list(std::ostringstream &         out,
                               const std::vector<Command> & commands,
                               const char *                 phase,
                               const KernelCorpus &         corpus,
                               const std::string &          target) {
    for (const Command & command : commands) {
        out << phase << ' ' << command.ordinal << ' ' << kernel_name(corpus, target, command.kernel.kernel_id);
        if (!command.dependencies.empty()) {
            out << " deps=";
            for (size_t i = 0; i < command.dependencies.size(); ++i) {
                if (i > 0) {
                    out << ',';
                }
                out << command.dependencies[i];
            }
        }
        out << '\n';
    }
}

static std::string format_kernel_list(const CommandProgram & program,
                                      const KernelCorpus &   corpus,
                                      const std::string &    target) {
    std::ostringstream out;
    append_kernel_list(out, program.initialization_commands, "init", corpus, target);
    append_kernel_list(out, program.commands, "main", corpus, target);
    return out.str();
}

static void append_dot_nodes(std::ostringstream &         out,
                             const std::vector<Command> & commands,
                             const char *                 prefix,
                             const char *                 phase,
                             const KernelCorpus &         corpus,
                             const std::string &          target) {
    for (const Command & command : commands) {
        const std::string name = kernel_name(corpus, target, command.kernel.kernel_id);
        out << "  " << prefix << command.ordinal << " [label=\"" << phase << ' ' << command.ordinal << "\\n"
            << dot_escape(name) << "\"];\n";
    }
}

static std::string format_kernel_dot(const CommandProgram & program,
                                     const KernelCorpus &   corpus,
                                     const std::string &    target) {
    std::ostringstream out;
    out << "digraph hrx_kernel_invocations {\n";
    out << "  rankdir=LR;\n";
    append_dot_nodes(out, program.initialization_commands, "init", "init", corpus, target);
    append_dot_nodes(out, program.commands, "main", "main", corpus, target);
    for (const Command & command : program.commands) {
        for (const uint32_t dependency : command.dependencies) {
            out << "  main" << dependency << " -> main" << command.ordinal << ";\n";
        }
    }
    out << "}\n";
    return out.str();
}

static void append_json_command_list(std::ostringstream &         out,
                                     const std::vector<Command> & commands,
                                     const char *                 phase,
                                     const KernelCorpus &         corpus,
                                     const std::string &          target,
                                     const char *                 indent) {
    for (size_t command_index = 0; command_index < commands.size(); ++command_index) {
        const Command & command = commands[command_index];
        if (command_index > 0) {
            out << ",\n";
        }
        const std::string name = kernel_name(corpus, target, command.kernel.kernel_id);
        out << indent << "{\n";
        out << indent << "  \"phase\": \"" << phase << "\",\n";
        out << indent << "  \"ordinal\": " << command.ordinal << ",\n";
        out << indent << "  \"kind\": \"" << json_escape(command_kind_name(command.kind)) << "\",\n";
        out << indent << "  \"kernel_id\": " << command.kernel.kernel_id << ",\n";
        out << indent << "  \"kernel\": \"" << json_escape(name) << "\",\n";
        out << indent << "  \"integer_parameters\": ";
        append_json_integer_map(out, command.kernel.integer_parameters);
        out << ",\n";
        out << indent << "  \"compile_parameters\": ";
        append_json_string_map(out, command.kernel.compile_parameters);
        out << ",\n";
        out << indent << "  \"dependencies\": [";
        for (size_t i = 0; i < command.dependencies.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << command.dependencies[i];
        }
        out << "],\n";
        out << indent << "  \"bindings\": [\n";
        for (size_t binding_index = 0; binding_index < command.bindings.size(); ++binding_index) {
            const CommandBinding & binding = command.bindings[binding_index];
            if (binding_index > 0) {
                out << ",\n";
            }
            out << indent << "    {\n";
            out << indent << "      \"index\": " << binding_index << ",\n";
            out << indent << "      \"name\": \"" << json_escape(binding.name) << "\",\n";
            out << indent << "      \"value\": " << binding.value.value << ",\n";
            out << indent << "      \"origin\": \"" << json_escape(command_binding_origin_name(binding.origin)) << "\",\n";
            out << indent << "      \"access\": \"" << json_escape(resource_access_name(binding.access)) << "\",\n";
            out << indent << "      \"offset\": " << binding.offset << ",\n";
            out << indent << "      \"length\": " << binding.length << "\n";
            out << indent << "    }";
        }
        out << '\n' << indent << "  ]\n";
        out << indent << '}';
    }
}

static std::string format_kernel_json(const CommandProgram & program,
                                      const KernelCorpus &   corpus,
                                      const std::string &    target,
                                      const std::string &    command_shape,
                                      uint64_t               dump_id,
                                      const std::string &    shape_hash) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"ggml-hrx-command-program-v1\",\n";
    out << "  \"dump_id\": " << dump_id << ",\n";
    out << "  \"shape_hash\": \"" << json_escape(shape_hash) << "\",\n";
    out << "  \"target\": \"" << json_escape(target) << "\",\n";
    out << "  \"command_shape\": \"" << json_escape(command_shape) << "\",\n";
    out << "  \"transients\": {\n";
    out << "    \"arena_size\": " << program.transients.arena_size << ",\n";
    out << "    \"arena_alignment\": " << program.transients.arena_alignment << ",\n";
    out << "    \"allocations\": [\n";
    for (size_t i = 0; i < program.transients.allocations.size(); ++i) {
        const TransientAllocation & allocation = program.transients.allocations[i];
        if (i > 0) {
            out << ",\n";
        }
        out << "      {\"value\": " << allocation.value.value << ", \"size\": " << allocation.size
            << ", \"alignment\": " << allocation.alignment << ", \"arena_offset\": " << allocation.arena_offset
            << '}';
    }
    out << "\n    ]\n";
    out << "  },\n";
    out << "  \"completion_counters\": {\n";
    out << "    \"arena_offset\": " << program.completion_counters.arena_offset << ",\n";
    out << "    \"byte_count\": " << program.completion_counters.byte_count << ",\n";
    out << "    \"count\": " << program.completion_counters.count << "\n";
    out << "  },\n";
    out << "  \"activation_publications\": [\n";
    for (size_t i = 0; i < program.activation_publication_diagnostics.size(); ++i) {
        const CommandPlanActivationPublicationDiagnostic & diagnostic =
            program.activation_publication_diagnostics[i];
        if (i > 0) {
            out << ",\n";
        }
        out << "    {\"source_value\": " << diagnostic.source_value.value
            << ", \"alternate_value\": " << diagnostic.alternate_value.value
            << ", \"consumer_value\": " << diagnostic.consumer_value.value
            << ", \"requested_format\": \"" << json_escape(diagnostic.requested_format)
            << "\", \"publication_name\": \"" << json_escape(diagnostic.publication_name)
            << "\", \"publication_stage\": \"" << json_escape(diagnostic.publication_stage)
            << "\", \"alias_depth\": " << diagnostic.alias_depth
            << ", \"mixed_fanout\": " << (diagnostic.mixed_fanout ? "true" : "false")
            << ", \"fallback_reason\": \"" << json_escape(diagnostic.fallback_reason) << "\"}";
    }
    out << "\n  ],\n";
    out << "  \"commands\": [\n";
    bool wrote_any = false;
    if (!program.initialization_commands.empty()) {
        append_json_command_list(out, program.initialization_commands, "init", corpus, target, "    ");
        wrote_any = true;
    }
    if (!program.commands.empty()) {
        if (wrote_any) {
            out << ",\n";
        }
        append_json_command_list(out, program.commands, "main", corpus, target, "    ");
    }
    out << "\n  ]\n";
    out << "}\n";
    return out.str();
}

static void write_file(const std::filesystem::path & path, const std::string & contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot create " + path.string());
    }
    output << contents;
    if (contents.empty() || contents.back() != '\n') {
        output << '\n';
    }
}

}  // namespace

Status dump_command_program_kernels_if_requested(const CommandProgram & program,
                                                 const KernelCorpus &   corpus,
                                                 const std::string &    target,
                                                 const std::string &    command_shape) {
    Status       status;
    const char * directory_value = std::getenv(kDumpDirectoryEnvironment);
    if (directory_value == nullptr || directory_value[0] == '\0' ||
        (directory_value[0] == '0' && directory_value[1] == '\0')) {
        return status;
    }

    static std::mutex                      mutex;
    static std::unordered_set<std::string> dumped_shapes;
    static std::atomic<uint64_t>           sequence{ 0 };

    const std::filesystem::path root_directory(directory_value);
    const std::string           dump_key = root_directory.string() + '\n' + target + '\n' + command_shape;
    uint64_t                    dump_id  = 0;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!dumped_shapes.insert(dump_key).second) {
            return status;
        }
        dump_id = sequence.fetch_add(1);
    }

    try {
        const uint64_t     shape_hash = hash_string(dump_key);
        const std::string  shape_hex  = hex_u64(shape_hash);
        std::ostringstream name;
        name << "program-" << dump_id << "-shape-" << shape_hex;
        const std::filesystem::path directory = root_directory / name.str();
        std::filesystem::create_directories(directory);
        write_file(directory / "kernels.txt", format_kernel_list(program, corpus, target));
        write_file(directory / "kernels.dot", format_kernel_dot(program, corpus, target));
        write_file(directory / "program.json", format_kernel_json(program, corpus, target, command_shape, dump_id, shape_hex));
    } catch (const std::exception & error) {
        status.log("failed to write HRX command program kernel dump: %s", error.what());
    }
    return status;
}

}  // namespace ggml::hrx
