#pragma once

#include "../dispatch-registry.h"

#include <array>
#include <cstddef>

namespace ggml::hrx {

enum class CommonActivationPublicationFormat {
    None,
    F16Row,
    F16K16Major,
    Q8_1X4,
    SymmetricI4K32,
};

enum class CommonActivationConsumerTraversal {
    Direct,
    FullRangeOrderedAliases,
};

enum class CommonActivationPublicationFallbackReason {
    None,
    MissingGraphIndex,
    NoEnabledCandidates,
    NoQualifiedConsumer,
    IncompatibleAlias,
};

struct CommonActivationPublicationCapabilities {
    uint32_t formats = DispatchActivationInputNone;
    uint32_t alias_traversal_formats = DispatchActivationInputNone;
    uint32_t producer_requirements = DispatchActivationProducerRequirementNone;
    uint32_t co_publication_formats = DispatchActivationInputNone;
    uint32_t consumer_sources = DispatchActivationConsumerSourceAll;
    uint32_t required_consumer_traits = DispatchActivationConsumerTraitNone;
};

struct CommonActivationPublicationDemand {
    CommonActivationPublicationFormat format         = CommonActivationPublicationFormat::None;
    const Value *                     consumer_value = nullptr;
    const GraphNode *                 consumer       = nullptr;
    size_t                            alias_depth     = 0;
    bool                              mixed_fanout    = false;
    CommonActivationPublicationFallbackReason fallback_reason = CommonActivationPublicationFallbackReason::None;

    bool matched() const {
        return format != CommonActivationPublicationFormat::None && consumer_value != nullptr && consumer != nullptr;
    }
};

struct CommonActivationPublicationPlan {
    std::array<CommonActivationPublicationDemand, 4> publications;
    size_t                                           publication_count = 0;
    CommonActivationPublicationFallbackReason       fallback_reason =
        CommonActivationPublicationFallbackReason::None;

    bool matched() const { return publication_count != 0; }

    const CommonActivationPublicationDemand & preferred() const { return publications.front(); }

    const CommonActivationPublicationDemand * find(CommonActivationPublicationFormat format) const {
        for (size_t i = 0; i < publication_count; ++i) {
            if (publications[i].format == format) {
                return &publications[i];
            }
        }
        return nullptr;
    }
};

struct CommonActivationPublication {
    CommonActivationPublicationFormat format = CommonActivationPublicationFormat::None;
    ValueId                           source_value;
    ValueId                           alternate_value;
    size_t                            byte_count    = 0;
    size_t                            payload_bytes = 0;
    size_t                            scales_offset = 0;
    size_t                            scales_bytes  = 0;
    size_t                            sums_offset   = 0;
    size_t                            sums_bytes    = 0;
    ValueId                           consumer_value;
    size_t                            alias_depth  = 0;
    bool                              mixed_fanout = false;

    bool matched() const {
        return format != CommonActivationPublicationFormat::None && source_value.value >= 0 &&
               alternate_value.value >= 0 && byte_count > 0;
    }

    DispatchBinding binding() const { return { alternate_value, 0, byte_count }; }

    DispatchBinding binding(size_t offset, size_t length) const { return { alternate_value, offset, length }; }
};

const char * common_activation_publication_format_name(CommonActivationPublicationFormat format);

const char * common_activation_publication_fallback_reason_name(CommonActivationPublicationFallbackReason reason);

CommonActivationPublicationPlan common_select_activation_publication_plan(
    const DispatchMatchContext &                    context,
    const Value &                                   produced_value,
    const CommonActivationPublicationCapabilities & capabilities);

bool common_reserve_activation_publication(const DispatchMatchContext &              context,
                                           DispatchMatch &                           match,
                                           const Value &                             produced_value,
                                           const CommonActivationPublicationDemand & demand,
                                           const char *                              transient_name,
                                           CommonActivationPublication &             publication);

bool common_reserve_private_activation_output(const DispatchMatchContext &        context,
                                              DispatchMatch &                     match,
                                              const Value &                       produced_value,
                                              CommonActivationPublicationFormat   format,
                                              const char *                        transient_name,
                                              CommonActivationPublication &       publication);

// A metadata conflict is appended to match.status and invalidates the entire match; callers must not fall back.
bool common_append_activation_publication(DispatchMatch &                     match,
                                          const CommonActivationPublication & publication,
                                          const char *                        metadata_name);

}  // namespace ggml::hrx
