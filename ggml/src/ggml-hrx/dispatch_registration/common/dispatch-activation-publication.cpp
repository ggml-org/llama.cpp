#include "dispatch-activation-publication.h"

#include "dispatch/command-plan.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace ggml::hrx {
namespace {

static bool is_ordered_alias_node(const Graph & graph, const GraphNode & node, const Value & input) {
    if ((node.op != GGML_OP_VIEW && node.op != GGML_OP_RESHAPE) || node.inputs.size() != 1 ||
        node.inputs.front() != input.id || !is_layout_alias_node(graph, node)) {
        return false;
    }
    const Value * output = graph.values().find(node.output);
    return output != nullptr && output->alias_source == input.id && same_full_ordered_value_range(input, *output);
}

static bool publication_values_compatible(CommonActivationPublicationFormat format,
                                          const Value &                     source,
                                          const Value &                     candidate) {
    if (format == CommonActivationPublicationFormat::F16Row) {
        return same_full_ordered_value_range(source, candidate);
    }
    if (format == CommonActivationPublicationFormat::Q8_1X4) {
        return same_q8_1_x4_geometry(source, candidate);
    }
    return same_packed_row_geometry(source, candidate);
}

static uint32_t publication_format_mask(CommonActivationPublicationFormat format) {
    switch (format) {
        case CommonActivationPublicationFormat::F16Row:
            return DispatchActivationInputF16Row;
        case CommonActivationPublicationFormat::F16K16Major:
            return DispatchActivationInputF16K16Major;
        case CommonActivationPublicationFormat::Q8_1X4:
            return DispatchActivationInputQ8_1X4;
        case CommonActivationPublicationFormat::SymmetricI4K32:
            return DispatchActivationInputSymmetricI4K32;
        case CommonActivationPublicationFormat::None:
            return DispatchActivationInputNone;
    }
    return DispatchActivationInputNone;
}

static CommonActivationPublicationFormat publication_format_from_mask(uint32_t format) {
    switch (format) {
        case DispatchActivationInputF16Row:
            return CommonActivationPublicationFormat::F16Row;
        case DispatchActivationInputF16K16Major:
            return CommonActivationPublicationFormat::F16K16Major;
        case DispatchActivationInputQ8_1X4:
            return CommonActivationPublicationFormat::Q8_1X4;
        case DispatchActivationInputSymmetricI4K32:
            return CommonActivationPublicationFormat::SymmetricI4K32;
        default:
            return CommonActivationPublicationFormat::None;
    }
}

static int publication_use_priority(DispatchActivationConsumerUse use) {
    switch (use) {
        case DispatchActivationConsumerUse::BandwidthLimited:
            return 400;
        case DispatchActivationConsumerUse::Tiled:
            return 300;
        case DispatchActivationConsumerUse::LowRow:
            return 250;
        case DispatchActivationConsumerUse::Row:
            return 200;
    }
    return 0;
}

static int publication_format_priority(uint32_t format) {
    switch (format) {
        case DispatchActivationInputQ8_1X4:
            return 40;
        case DispatchActivationInputF16K16Major:
            return 30;
        case DispatchActivationInputSymmetricI4K32:
            return 20;
        case DispatchActivationInputF16Row:
            return 10;
        default:
            return 0;
    }
}

static uint32_t publication_consumer_source_mask(DispatchSource source) {
    switch (source) {
        case DispatchSource::Common:
            return DispatchActivationConsumerSourceCommon;
        case DispatchSource::Llm:
            return DispatchActivationConsumerSourceLlm;
        case DispatchSource::Qwen:
            return DispatchActivationConsumerSourceQwen;
    }
    return 0;
}

static bool publication_has_mixed_fanout(const Graph &     graph,
                                         const Value &     produced_value,
                                         const Value &     consumer_value,
                                         const GraphNode & selected_consumer) {
    const Value *     current          = &consumer_value;
    const GraphNode * allowed_consumer = &selected_consumer;
    for (size_t depth = 0; depth <= graph.values().size(); ++depth) {
        for (const GraphNode * consumer : graph.index().consumers(current->id)) {
            if (consumer != nullptr && consumer != allowed_consumer) {
                return true;
            }
        }
        if (current->id == produced_value.id) {
            return false;
        }
        const GraphNode * alias = graph.index().producer(current->id);
        if (alias == nullptr || alias->inputs.size() != 1) {
            return true;
        }
        const Value * source = graph.values().find(alias->inputs.front());
        if (source == nullptr) {
            return true;
        }
        allowed_consumer = alias;
        current          = source;
    }
    return true;
}

static size_t align_up(size_t value, size_t alignment) {
    return value > std::numeric_limits<size_t>::max() - (alignment - 1) ?
               0 :
               (value + alignment - 1) / alignment * alignment;
}

static size_t add_and_align(size_t value, size_t increment, size_t alignment) {
    return value > std::numeric_limits<size_t>::max() - increment ? 0 : align_up(value + increment, alignment);
}

static bool checked_element_count(const Value & value, size_t & element_count) {
    if (value.type != GGML_TYPE_F32 || !value.contiguous || value.element_count <= 0 || value.ne[0] <= 0 ||
        value.element_count % value.ne[0] != 0) {
        return false;
    }
    element_count = static_cast<size_t>(value.element_count);
    return static_cast<int64_t>(element_count) == value.element_count;
}

static bool set_publication_layout(CommonActivationPublicationFormat format,
                                   const Value &                     value,
                                   CommonActivationPublication &     publication) {
    size_t element_count = 0;
    if (!checked_element_count(value, element_count)) {
        return false;
    }

    switch (format) {
        case CommonActivationPublicationFormat::F16Row:
        case CommonActivationPublicationFormat::F16K16Major:
            if (element_count > std::numeric_limits<size_t>::max() / sizeof(ggml_fp16_t)) {
                return false;
            }
            publication.byte_count    = element_count * sizeof(ggml_fp16_t);
            publication.payload_bytes = publication.byte_count;
            return true;
        case CommonActivationPublicationFormat::Q8_1X4:
            {
                if (value.ne[0] % 128 != 0) {
                    return false;
                }
                const size_t row_count = element_count / static_cast<size_t>(value.ne[0]);
                const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_1, value.ne[0]);
                if (row_count > std::numeric_limits<size_t>::max() / row_bytes) {
                    return false;
                }
                publication.byte_count    = row_count * row_bytes;
                publication.payload_bytes = publication.byte_count;
                return publication.byte_count > 0;
            }
        case CommonActivationPublicationFormat::SymmetricI4K32:
            {
                if (value.ne[0] % 64 != 0 || element_count % 8 != 0) {
                    return false;
                }
                const size_t payload_bytes  = element_count / 2;
                const size_t metadata_bytes = element_count / 8;
                const size_t scales_offset  = align_up(payload_bytes, 256);
                const size_t sums_offset = scales_offset == 0 ? 0 : add_and_align(scales_offset, metadata_bytes, 256);
                const size_t total_bytes = sums_offset == 0 ? 0 : add_and_align(sums_offset, metadata_bytes, 256);
                if (total_bytes == 0) {
                    return false;
                }
                publication.byte_count    = total_bytes;
                publication.payload_bytes = payload_bytes;
                publication.scales_offset = scales_offset;
                publication.scales_bytes  = metadata_bytes;
                publication.sums_offset   = sums_offset;
                publication.sums_bytes    = metadata_bytes;
                return true;
            }
        case CommonActivationPublicationFormat::None:
            return false;
    }
    return false;
}

static bool reserve_activation_output(const DispatchMatchContext &      context,
                                      DispatchMatch &                   match,
                                      const Value &                     layout_value,
                                      const char *                      transient_name,
                                      CommonActivationPublication &     publication) {
    if (transient_name == nullptr || !set_publication_layout(publication.format, layout_value, publication)) {
        publication = {};
        return false;
    }

    const size_t value_offset = match.transients.size() + match.completion_counter_requests.size();
    if (context.next_plan_value.value < 0 || value_offset > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        context.next_plan_value.value > std::numeric_limits<int32_t>::max() - static_cast<int32_t>(value_offset)) {
        publication = {};
        return false;
    }
    publication.alternate_value = ValueId(context.next_plan_value.value + static_cast<int32_t>(value_offset));
    match.transients.push_back({ publication.alternate_value, transient_name, publication.byte_count, 256 });
    return true;
}

}  // namespace

const char * common_activation_publication_format_name(CommonActivationPublicationFormat format) {
    switch (format) {
        case CommonActivationPublicationFormat::F16Row:
            return "f16-row";
        case CommonActivationPublicationFormat::F16K16Major:
            return "f16-k16-major";
        case CommonActivationPublicationFormat::Q8_1X4:
            return "q8-1-x4";
        case CommonActivationPublicationFormat::SymmetricI4K32:
            return "symmetric-i4-k32";
        case CommonActivationPublicationFormat::None:
            return "none";
    }
    return "none";
}

const char * common_activation_publication_fallback_reason_name(CommonActivationPublicationFallbackReason reason) {
    switch (reason) {
        case CommonActivationPublicationFallbackReason::None:
            return "none";
        case CommonActivationPublicationFallbackReason::MissingGraphIndex:
            return "missing-graph-index";
        case CommonActivationPublicationFallbackReason::NoEnabledCandidates:
            return "no-enabled-candidates";
        case CommonActivationPublicationFallbackReason::NoQualifiedConsumer:
            return "no-qualified-consumer";
        case CommonActivationPublicationFallbackReason::IncompatibleAlias:
            return "incompatible-alias";
    }
    return "unknown";
}

CommonActivationPublicationPlan common_select_activation_publication_plan(
    const DispatchMatchContext &                    context,
    const Value &                                   produced_value,
    const CommonActivationPublicationCapabilities & capabilities) {
    CommonActivationPublicationPlan result;
    if (!context.graph.has_index()) {
        result.fallback_reason = CommonActivationPublicationFallbackReason::MissingGraphIndex;
        return result;
    }
    if (capabilities.formats == DispatchActivationInputNone) {
        result.fallback_reason = CommonActivationPublicationFallbackReason::NoEnabledCandidates;
        return result;
    }
    if (context.registry == nullptr) {
        result.fallback_reason = CommonActivationPublicationFallbackReason::NoQualifiedConsumer;
        return result;
    }

    struct TraversedValue {
        const Value * value = nullptr;
        size_t depth = 0;
        uint32_t eligible_formats = DispatchActivationInputNone;
    };
    struct Selection {
        CommonActivationPublicationDemand demand;
        int priority = 0;
    };
    std::vector<TraversedValue> values = { { &produced_value, 0, capabilities.formats } };
    std::array<Selection, 4> best_by_format;
    bool incompatible_alias = false;
    for (size_t value_index = 0; value_index < values.size(); ++value_index) {
        const Value & value = *values[value_index].value;
        for (const GraphNode * consumer : context.graph.index().consumers(value.id)) {
            if (consumer == nullptr) {
                continue;
            }
            const auto & classifiers = context.registry->activation_consumers_for_root(consumer->op);
            for (const DispatchActivationConsumerRegistration & classifier : classifiers) {
                if ((capabilities.consumer_sources & publication_consumer_source_mask(classifier.source)) == 0 ||
                    (classifier.traits & capabilities.required_consumer_traits) !=
                        capabilities.required_consumer_traits ||
                    (classifier.producer_requirements & capabilities.producer_requirements) !=
                        classifier.producer_requirements ||
                    !classifier.accepts(context, *consumer, value)) {
                    continue;
                }
                uint32_t accepted = classifier.accepted_formats & values[value_index].eligible_formats;
                while (accepted != 0) {
                    const uint32_t format = accepted & (~accepted + 1);
                    accepted &= ~format;
                    const int priority = publication_use_priority(classifier.use) * 100 +
                                         publication_format_priority(format);
                    const size_t format_index = static_cast<size_t>(publication_format_from_mask(format)) - 1;
                    Selection & best = best_by_format[format_index];
                    if (priority > best.priority) {
                        best.priority = priority;
                        best.demand = { publication_format_from_mask(format),
                                        &value,
                                        consumer,
                                        values[value_index].depth,
                                        publication_has_mixed_fanout(context.graph, produced_value, value, *consumer),
                                        CommonActivationPublicationFallbackReason::None };
                    }
                }
            }

            if ((values[value_index].eligible_formats & capabilities.alias_traversal_formats) == 0 ||
                values[value_index].depth >= context.graph.values().size() ||
                !is_ordered_alias_node(context.graph, *consumer, value)) {
                continue;
            }
            const Value * alias = context.graph.values().find(consumer->output);
            if (alias == nullptr) {
                continue;
            }
            uint32_t alias_formats = DispatchActivationInputNone;
            for (uint32_t format = DispatchActivationInputF16Row;
                 format <= DispatchActivationInputSymmetricI4K32; format <<= 1) {
                if ((values[value_index].eligible_formats & capabilities.alias_traversal_formats & format) != 0 &&
                    publication_values_compatible(publication_format_from_mask(format), produced_value, *alias)) {
                    alias_formats |= format;
                }
            }
            if (alias_formats == DispatchActivationInputNone) {
                incompatible_alias = true;
                continue;
            }
            TraversedValue * existing = nullptr;
            for (TraversedValue & traversed : values) {
                if (traversed.value->id == alias->id) {
                    existing = &traversed;
                    break;
                }
            }
            if (existing == nullptr) {
                values.push_back({ alias, values[value_index].depth + 1, alias_formats });
            } else {
                existing->eligible_formats |= alias_formats;
            }
        }
    }
    std::vector<Selection> selections;
    for (const Selection & selection : best_by_format) {
        if (selection.priority != 0) {
            selections.push_back(selection);
        }
    }
    std::stable_sort(selections.begin(), selections.end(), [](const Selection & lhs, const Selection & rhs) {
        return lhs.priority > rhs.priority;
    });
    if (selections.empty()) {
        result.fallback_reason = incompatible_alias ? CommonActivationPublicationFallbackReason::IncompatibleAlias :
                                                      CommonActivationPublicationFallbackReason::NoQualifiedConsumer;
        return result;
    }

    result.publications[result.publication_count++] = selections.front().demand;
    const CommonActivationPublicationDemand & preferred = selections.front().demand;
    const uint32_t preferred_format = publication_format_mask(preferred.format);
    if ((capabilities.co_publication_formats & preferred_format) != 0) {
        static constexpr uint32_t kFormatPriority[] = {
            DispatchActivationInputQ8_1X4,
            DispatchActivationInputF16K16Major,
            DispatchActivationInputSymmetricI4K32,
            DispatchActivationInputF16Row,
        };
        for (uint32_t format : kFormatPriority) {
            if (format == preferred_format || (capabilities.co_publication_formats & format) == 0 ||
                result.publication_count >= result.publications.size()) {
                continue;
            }
            const CommonActivationPublicationFormat publication_format = publication_format_from_mask(format);
            const auto selection = std::find_if(
                selections.begin(), selections.end(), [publication_format](const Selection & candidate) {
                    return candidate.demand.format == publication_format;
                });
            if (selection != selections.end()) {
                result.publications[result.publication_count++] = selection->demand;
            } else if (publication_values_compatible(publication_format, produced_value, *preferred.consumer_value)) {
                CommonActivationPublicationDemand companion = preferred;
                companion.format = publication_format;
                result.publications[result.publication_count++] = companion;
            }
        }
    }
    return result;
}

bool common_reserve_activation_publication(const DispatchMatchContext &              context,
                                           DispatchMatch &                           match,
                                           const Value &                             produced_value,
                                           const CommonActivationPublicationDemand & demand,
                                           const char *                              transient_name,
                                           CommonActivationPublication &             publication) {
    publication = {};
    if (!demand.matched() || transient_name == nullptr ||
        !publication_values_compatible(demand.format, produced_value, *demand.consumer_value)) {
        return false;
    }

    publication.format       = demand.format;
    publication.source_value = produced_value.id;
    publication.consumer_value = demand.consumer_value->id;
    publication.alias_depth    = demand.alias_depth;
    publication.mixed_fanout   = demand.mixed_fanout;
    return reserve_activation_output(context, match, *demand.consumer_value, transient_name, publication);
}

bool common_reserve_private_activation_output(const DispatchMatchContext &      context,
                                              DispatchMatch &                   match,
                                              const Value &                     produced_value,
                                              CommonActivationPublicationFormat format,
                                              const char *                      transient_name,
                                              CommonActivationPublication &     publication) {
    publication = {};
    publication.format         = format;
    publication.source_value   = produced_value.id;
    publication.consumer_value = produced_value.id;
    return reserve_activation_output(context, match, produced_value, transient_name, publication);
}

bool common_append_activation_publication(DispatchMatch &                     match,
                                          const CommonActivationPublication & publication,
                                          const char *                        metadata_name) {
    if (!publication.matched() || metadata_name == nullptr) {
        return false;
    }

    Status status;
    bool   recorded = false;
    switch (publication.format) {
        case CommonActivationPublicationFormat::F16Row:
            recorded = match.metadata.append_alternate_value({ publication.source_value, publication.alternate_value,
                                                               GGML_TYPE_F16, publication.byte_count, metadata_name },
                                                             status);
            break;
        case CommonActivationPublicationFormat::F16K16Major:
            recorded = match.metadata.append_generated_resource({ publication.source_value,
                                                                  GeneratedResourceRole::F16K16Major,
                                                                  publication.alternate_value,
                                                                  publication.byte_count,
                                                                  {} },
                                                                status);
            break;
        case CommonActivationPublicationFormat::Q8_1X4:
            recorded = match.metadata.append_alternate_value({ publication.source_value, publication.alternate_value,
                                                               GGML_TYPE_Q8_1, publication.byte_count, metadata_name },
                                                             status);
            break;
        case CommonActivationPublicationFormat::SymmetricI4K32:
            recorded = match.metadata.append_alternate_value({ publication.source_value, publication.alternate_value,
                                                               GGML_TYPE_COUNT, publication.byte_count, metadata_name },
                                                             status);
            break;
        case CommonActivationPublicationFormat::None:
            break;
    }
    if (!recorded) {
        match.status.append(status);
    } else {
        match.metadata.append_activation_publication_diagnostic({ publication.source_value,
                                                                  publication.alternate_value,
                                                                  publication.consumer_value,
                                                                  common_activation_publication_format_name(
                                                                      publication.format),
                                                                  metadata_name,
                                                                  "published",
                                                                  "none",
                                                                  publication.alias_depth,
                                                                  publication.mixed_fanout });
    }
    return recorded;
}

}  // namespace ggml::hrx
