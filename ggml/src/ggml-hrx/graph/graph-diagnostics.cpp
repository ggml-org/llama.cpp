#include "graph-diagnostics.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <atomic>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace ggml::hrx {
namespace {

using json = nlohmann::ordered_json;

const char * value_kind_name(ValueKind kind) {
    switch (kind) {
        case ValueKind::External:
            return "external";
        case ValueKind::Transient:
            return "transient";
    }
    return "unknown";
}

bool parse_value_kind(const std::string & name, ValueKind & kind) {
    if (name == "external") {
        kind = ValueKind::External;
        return true;
    }
    if (name == "transient") {
        kind = ValueKind::Transient;
        return true;
    }
    return false;
}

const char * match_kind_name(DispatchMatchKind kind) {
    switch (kind) {
        case DispatchMatchKind::Fused:
            return "fused";
        case DispatchMatchKind::SingleOp:
            return "single_op";
    }
    return "unknown";
}

const char * dispatch_source_name(DispatchSource source) {
    switch (source) {
        case DispatchSource::Common:
            return "common";
        case DispatchSource::Llm:
            return "llm";
        case DispatchSource::Qwen:
            return "qwen";
    }
    return "unknown";
}

json dims_json(const std::array<int64_t, GGML_MAX_DIMS> & values) {
    json result = json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.push_back(values[i]);
    }
    return result;
}

json strides_json(const std::array<size_t, GGML_MAX_DIMS> & values) {
    json result = json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        result.push_back(values[i]);
    }
    return result;
}

Status read_dims(const json & item, const char * name, std::array<int64_t, GGML_MAX_DIMS> & values) {
    Status status;
    if (!item.contains(name) || !item[name].is_array() || item[name].size() != GGML_MAX_DIMS) {
        status.log("snapshot array %s must contain %d values", name, GGML_MAX_DIMS);
        return status;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        values[i] = item[name][i].get<int64_t>();
    }
    return status;
}

Status read_strides(const json & item, const char * name, std::array<size_t, GGML_MAX_DIMS> & values) {
    Status status;
    if (!item.contains(name) || !item[name].is_array() || item[name].size() != GGML_MAX_DIMS) {
        status.log("snapshot array %s must contain %d values", name, GGML_MAX_DIMS);
        return status;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        values[i] = item[name][i].get<size_t>();
    }
    return status;
}

json op_params_json(const OpParams & params) {
    return std::visit(
        [](const auto & value) -> json {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                return {
                    { "kind", "none" }
                };
            } else if constexpr (std::is_same_v<T, RmsNormParams>) {
                return {
                    { "kind", "rms_norm" },
                    { "eps",  value.eps  }
                };
            } else if constexpr (std::is_same_v<T, FlashAttnExtParams>) {
                return {
                    { "kind",          "flash_attn_ext"             },
                    { "scale",         value.scale                  },
                    { "max_bias",      value.max_bias               },
                    { "logit_softcap", value.logit_softcap          },
                    { "prec",          static_cast<int>(value.prec) },
                };
            } else if constexpr (std::is_same_v<T, SoftMaxParams>) {
                return {
                    { "kind",     "soft_max"     },
                    { "scale",    value.scale    },
                    { "max_bias", value.max_bias }
                };
            } else if constexpr (std::is_same_v<T, ArgsortParams>) {
                return {
                    { "kind",  "argsort"                     },
                    { "order", static_cast<int>(value.order) }
                };
            } else if constexpr (std::is_same_v<T, ClampParams>) {
                return {
                    { "kind", "clamp"   },
                    { "min",  value.min },
                    { "max",  value.max }
                };
            } else if constexpr (std::is_same_v<T, GluParams>) {
                return {
                    { "kind", "glu"                      },
                    { "op",   static_cast<int>(value.op) }
                };
            } else if constexpr (std::is_same_v<T, RopeParams>) {
                return {
                    { "kind",        "rope"            },
                    { "n_dims",      value.n_dims      },
                    { "mode",        value.mode        },
                    { "n_ctx_orig",  value.n_ctx_orig  },
                    { "freq_base",   value.freq_base   },
                    { "freq_scale",  value.freq_scale  },
                    { "ext_factor",  value.ext_factor  },
                    { "attn_factor", value.attn_factor },
                    { "beta_fast",   value.beta_fast   },
                    { "beta_slow",   value.beta_slow   },
                };
            }
        },
        params);
}

OpParams parse_op_params(const json & item) {
    const std::string kind = item.value("kind", "none");
    if (kind == "rms_norm") {
        return RmsNormParams{ item.value("eps", 0.0f) };
    }
    if (kind == "flash_attn_ext") {
        return FlashAttnExtParams{
            item.value("scale", 0.0f),
            item.value("max_bias", 0.0f),
            item.value("logit_softcap", 0.0f),
            static_cast<ggml_prec>(item.value("prec", static_cast<int>(GGML_PREC_DEFAULT))),
        };
    }
    if (kind == "soft_max") {
        return SoftMaxParams{ item.value("scale", 0.0f), item.value("max_bias", 0.0f) };
    }
    if (kind == "argsort") {
        return ArgsortParams{ static_cast<ggml_sort_order>(
            item.value("order", static_cast<int>(GGML_SORT_ORDER_ASC))) };
    }
    if (kind == "clamp") {
        const float min = item.contains("min") && !item["min"].is_null() ? item["min"].get<float>() : 0.0f;
        const float max = item.contains("max") && !item["max"].is_null() ? item["max"].get<float>() :
                                                                           std::numeric_limits<float>::infinity();
        return ClampParams{ min, max };
    }
    if (kind == "glu") {
        return GluParams{ static_cast<ggml_glu_op>(item.value("op", static_cast<int>(GGML_GLU_OP_REGLU))) };
    }
    if (kind == "rope") {
        return RopeParams{
            item.value("n_dims", 0),         item.value("mode", 0),          item.value("n_ctx_orig", 0),
            item.value("freq_base", 0.0f),   item.value("freq_scale", 0.0f), item.value("ext_factor", 0.0f),
            item.value("attn_factor", 0.0f), item.value("beta_fast", 0.0f),  item.value("beta_slow", 0.0f),
        };
    }
    return std::monostate{};
}

json value_json(const Value & value) {
    return {
        { "id",                 value.id.value               },
        { "kind",               value_kind_name(value.kind)  },
        { "storage",            value.storage.value          },
        { "storage_root",       value.storage_root.value     },
        { "alias_source",       value.alias_source.value     },
        { "storage_offset",     value.storage_offset         },
        { "storage_byte_count", value.storage_byte_count     },
        { "type",               static_cast<int>(value.type) },
        { "type_name",          ggml_type_name(value.type)   },
        { "ne",                 dims_json(value.ne)          },
        { "nb",                 strides_json(value.nb)       },
        { "element_count",      value.element_count          },
        { "byte_count",         value.byte_count             },
        { "contiguous",         value.contiguous             },
    };
}

json storage_json(const ValueStorage & storage) {
    return {
        { "id",         storage.id.value   },
        { "root",       storage.root.value },
        { "byte_count", storage.byte_count },
    };
}

json node_json(const GraphNode & node) {
    json inputs = json::array();
    for (ValueId input : node.inputs) {
        inputs.push_back(input.value);
    }
    return {
        { "op",      static_cast<int>(node.op)   },
        { "op_name", ggml_op_name(node.op)       },
        { "output",  node.output.value           },
        { "inputs",  std::move(inputs)           },
        { "params",  op_params_json(node.params) },
    };
}

std::string value_summary(const Graph & graph, ValueId id) {
    std::ostringstream out;
    const Value *      value = graph.values().find(id);
    if (value == nullptr) {
        out << id.value << ":missing";
        return out.str();
    }
    out << id.value << ":" << ggml_type_name(value->type) << "[" << value->ne[0] << "," << value->ne[1] << ","
        << value->ne[2] << "," << value->ne[3] << "] " << value_kind_name(value->kind);
    if (value->alias_source.value >= 0) {
        out << " alias=" << value->alias_source.value << " storage_offset=" << value->storage_offset;
    }
    return out.str();
}

json attempts_json(const DispatchMatchDiagnostics & diagnostics) {
    json attempts = json::array();
    for (const DispatchRegistrationAttempt & attempt : diagnostics.attempts) {
        json covered = json::array();
        for (size_t node : attempt.covered_nodes) {
            covered.push_back(node);
        }
        attempts.push_back({
            { "name",          attempt.name                         },
            { "root_op",       static_cast<int>(attempt.root_op)    },
            { "root_op_name",  ggml_op_name(attempt.root_op)        },
            { "kind",          match_kind_name(attempt.kind)        },
            { "priority",      attempt.priority                     },
            { "source",        dispatch_source_name(attempt.source) },
            { "matched",       attempt.matched                      },
            { "covered_nodes", std::move(covered)                   },
            { "errors",        attempt.errors                       },
        });
    }
    return attempts;
}

void write_file(const std::filesystem::path & path, const std::string & contents) {
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

}  // namespace

std::string serialize_graph_snapshot_json(const Graph & graph, const std::string & target, uint64_t uid) {
    json values = json::array();
    for (const Value & value : graph.values().values()) {
        values.push_back(value_json(value));
    }
    json storages = json::array();
    for (const ValueStorage & storage : graph.values().storages()) {
        storages.push_back(storage_json(storage));
    }
    json nodes = json::array();
    for (const GraphNode & node : graph.nodes()) {
        nodes.push_back(node_json(node));
    }
    json root = {
        { "schema",   "ggml-hrx-graph-snapshot-v1" },
        { "uid",      uid                          },
        { "target",   target                       },
        { "storages", std::move(storages)          },
        { "values",   std::move(values)            },
        { "nodes",    std::move(nodes)             },
    };
    return root.dump(2);
}

std::string format_graph_snapshot_text(const Graph & graph, const std::string & target, uint64_t uid) {
    std::ostringstream out;
    out << "schema=ggml-hrx-graph-snapshot-v1\n";
    out << "uid=" << uid << "\ntarget=" << target << "\nvalues=" << graph.values().size()
        << "\nnodes=" << graph.nodes().size() << '\n';
    for (size_t i = 0; i < graph.nodes().size(); ++i) {
        const GraphNode & node = graph.nodes()[i];
        out << "node " << i << " " << ggml_op_name(node.op) << " output=" << value_summary(graph, node.output)
            << " inputs=[";
        for (size_t j = 0; j < node.inputs.size(); ++j) {
            if (j > 0) {
                out << ", ";
            }
            out << value_summary(graph, node.inputs[j]);
        }
        out << "] consumers=[";
        const std::vector<const GraphNode *> & consumers = graph.index().consumers(node.output);
        for (size_t j = 0; j < consumers.size(); ++j) {
            size_t consumer_index = 0;
            if (j > 0) {
                out << ", ";
            }
            if (graph.index().node_index(consumers[j], consumer_index)) {
                out << consumer_index << ":" << ggml_op_name(consumers[j]->op);
            }
        }
        out << "]\n";
    }
    return out.str();
}

GraphSnapshotLoadResult load_graph_snapshot_json(const std::string & contents) {
    GraphSnapshotLoadResult result;
    try {
        const json root = json::parse(contents);
        if (root.value("schema", "") != "ggml-hrx-graph-snapshot-v1") {
            result.status.log("unsupported graph snapshot schema");
            return result;
        }
        result.uid    = root.value("uid", 0ULL);
        result.target = root.value("target", "");

        ValueMap & values = result.graph.values();
        for (const json & item : root.at("storages")) {
            Status status = values.add_snapshot_storage({
                ValueStorageId(item.at("id").get<int32_t>()),
                ValueId(item.at("root").get<int32_t>()),
                item.at("byte_count").get<size_t>(),
            });
            if (!status.success()) {
                result.status.append(status);
                return result;
            }
        }
        for (const json & item : root.at("values")) {
            Value value = {
                ValueId(item.at("id").get<int32_t>()),
                ValueKind::External,
                ValueStorageId(item.at("storage").get<int32_t>()),
                ValueId(item.at("storage_root").get<int32_t>()),
                ValueId(item.at("alias_source").get<int32_t>()),
                item.at("storage_offset").get<size_t>(),
                item.at("storage_byte_count").get<size_t>(),
                static_cast<ggml_type>(item.at("type").get<int>()),
                {},
                {},
                item.at("element_count").get<int64_t>(),
                item.at("byte_count").get<size_t>(),
                item.at("contiguous").get<bool>(),
                nullptr,
                std::nullopt,
            };
            if (!parse_value_kind(item.at("kind").get<std::string>(), value.kind)) {
                result.status.log("snapshot value %d has unknown kind", value.id.value);
                return result;
            }
            Status dims_status = read_dims(item, "ne", value.ne);
            if (!dims_status.success()) {
                result.status.append(dims_status);
                return result;
            }
            Status strides_status = read_strides(item, "nb", value.nb);
            if (!strides_status.success()) {
                result.status.append(strides_status);
                return result;
            }
            Status status = values.add_snapshot_value(std::move(value));
            if (!status.success()) {
                result.status.append(status);
                return result;
            }
        }
        for (const json & item : root.at("nodes")) {
            std::vector<ValueId> inputs;
            for (const json & input : item.at("inputs")) {
                inputs.push_back(ValueId(input.get<int32_t>()));
            }
            GraphNode & node = result.graph.add_node(static_cast<ggml_op>(item.at("op").get<int>()),
                                                     ValueId(item.at("output").get<int32_t>()), std::move(inputs));
            node.params      = parse_op_params(item.at("params"));
        }
        result.status.append(result.graph.build_index());
    } catch (const std::exception & error) {
        result.status.log("failed to load graph snapshot: %s", error.what());
    }
    return result;
}

Status write_graph_snapshot(const std::filesystem::path & directory,
                            const Graph &                 graph,
                            const std::string &           target,
                            uint64_t                      uid) {
    Status status;
    try {
        static std::atomic<uint64_t> sequence{ 0 };
        const uint64_t               id = sequence.fetch_add(1);
        std::ostringstream           name;
        name << "graph-" << id << "-uid-" << uid << "-" << target << "-" << graph.nodes().size() << "-nodes";
        const std::filesystem::path base = directory / name.str();
        write_file(base.string() + ".json", serialize_graph_snapshot_json(graph, target, uid));
        write_file(base.string() + ".txt", format_graph_snapshot_text(graph, target, uid));
    } catch (const std::exception & error) {
        status.log("failed to write HRX graph snapshot: %s", error.what());
    }
    return status;
}

std::string format_schedule_diagnostics_text(const Graph &                       graph,
                                             const CommandPlan &                 plan,
                                             const DispatchScheduleDiagnostics & diagnostics) {
    std::ostringstream out;
    out << "valid=" << (plan.valid() ? "true" : "false") << '\n';
    for (const std::string & error : plan.status.errors()) {
        out << "error=" << error << '\n';
    }
    if (diagnostics.unsupported_node != nullptr) {
        out << "unsupported_node=" << diagnostics.unsupported_node_index << ":"
            << ggml_op_name(diagnostics.unsupported_node->op) << '\n';
        out << "unsupported_message=" << diagnostics.unsupported_message << '\n';
        out << "output=" << value_summary(graph, diagnostics.unsupported_node->output) << '\n';
        for (size_t i = 0; i < diagnostics.unsupported_node->inputs.size(); ++i) {
            out << "input" << i << "=" << value_summary(graph, diagnostics.unsupported_node->inputs[i]) << '\n';
        }
    }
    out << "matcher_attempts=" << diagnostics.match.attempts.size() << '\n';
    for (const DispatchRegistrationAttempt & attempt : diagnostics.match.attempts) {
        out << "attempt name=" << attempt.name << " kind=" << match_kind_name(attempt.kind)
            << " source=" << dispatch_source_name(attempt.source) << " priority=" << attempt.priority
            << " matched=" << (attempt.matched ? "true" : "false") << " covered=[";
        for (size_t i = 0; i < attempt.covered_nodes.size(); ++i) {
            if (i > 0) {
                out << ",";
            }
            out << attempt.covered_nodes[i];
        }
        out << "]\n";
        for (const std::string & error : attempt.errors) {
            out << "  error=" << error << '\n';
        }
    }
    return out.str();
}

std::string serialize_schedule_diagnostics_json(const Graph &                       graph,
                                                const CommandPlan &                 plan,
                                                const DispatchScheduleDiagnostics & diagnostics) {
    json errors = json::array();
    for (const std::string & error : plan.status.errors()) {
        errors.push_back(error);
    }
    json root = {
        { "schema",           "ggml-hrx-schedule-diagnostics-v1" },
        { "valid",            plan.valid()                       },
        { "errors",           std::move(errors)                  },
        { "matcher_attempts", attempts_json(diagnostics.match)   },
    };
    if (diagnostics.unsupported_node != nullptr) {
        json inputs = json::array();
        for (ValueId input : diagnostics.unsupported_node->inputs) {
            inputs.push_back(input.value);
        }
        root["unsupported_node"] = {
            { "index",   diagnostics.unsupported_node_index                 },
            { "op",      static_cast<int>(diagnostics.unsupported_node->op) },
            { "op_name", ggml_op_name(diagnostics.unsupported_node->op)     },
            { "output",  diagnostics.unsupported_node->output.value         },
            { "inputs",  std::move(inputs)                                  },
            { "message", diagnostics.unsupported_message                    },
        };
    }
    GGML_UNUSED(graph);
    return root.dump(2);
}

}  // namespace ggml::hrx
