#include "command-program-bindings.h"

#include <sstream>
#include <utility>

namespace ggml::hrx {
namespace {

static void mix_hash(uint64_t & hash, uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

}  // namespace

CommandProgramBindings CommandProgramBindings::from_value_map(const ValueMap & values) {
    std::vector<CommandProgramBinding> bindings;
    CommandProgramBindings             result;
    for (const ValueId id : values.external_value_ids()) {
        const Value * value = values.find(id);
        if (value == nullptr) {
            result.status.log("external value %d does not exist", id.value);
            continue;
        }
        const std::optional<ValueBufferBinding> buffer = values.resolve_buffer_binding(id);
        if (!buffer.has_value()) {
            result.status.log("external value %d is not bound", id.value);
            continue;
        }
        bindings.push_back({ value->id, buffer->buffer, buffer->offset, buffer->length, buffer->identity,
                             buffer->generation, buffer->capacity, buffer->host_data, buffer->weight });
    }
    return from_bindings(std::move(bindings), result.status);
}

CommandProgramBindings CommandProgramBindings::from_bindings(std::vector<CommandProgramBinding> bindings,
                                                             const Status &                     errors) {
    CommandProgramBindings result;
    result.status.append(errors);
    result.bindings_ = std::move(bindings);
    for (const CommandProgramBinding & binding : result.bindings_) {
        if (binding.buffer == nullptr && binding.host_data == nullptr) {
            result.status.log("external value %d has a null binding", binding.value.value);
        }
        if (binding.length == 0) {
            result.status.log("external value %d has an empty binding", binding.value.value);
        }
    }
    return result;
}

const CommandProgramBinding * CommandProgramBindings::find(ValueId value) const {
    for (const CommandProgramBinding & binding : bindings_) {
        if (binding.value == value) {
            return &binding;
        }
    }
    return nullptr;
}

CommandProgramBindingsHash command_program_bindings_hash(const CommandProgramBindings & bindings) {
    uint64_t hash = UINT64_C(1469598103934665603);
    mix_hash(hash, UINT64_C(0x6872782d62696e64));
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        mix_hash(hash, static_cast<uint64_t>(static_cast<int64_t>(binding.value.value)));
        mix_hash(hash, binding.host_data != nullptr ? 1 : 0);
        mix_hash(hash, binding.identity);
        mix_hash(hash, binding.generation);
        mix_hash(hash, static_cast<uint64_t>(binding.capacity));
        mix_hash(hash, static_cast<uint64_t>(binding.offset));
        mix_hash(hash, static_cast<uint64_t>(binding.length));
        mix_hash(hash, binding.weight ? 1 : 0);
    }
    mix_hash(hash, static_cast<uint64_t>(bindings.bindings().size()));
    return { hash };
}

CommandProgramBindingsFingerprint command_program_bindings_fingerprint(const CommandProgramBindings & bindings) {
    std::ostringstream out;
    out << "hrx-bindings-v1";
    for (const CommandProgramBinding & binding : bindings.bindings()) {
        out << "|value=" << binding.value.value << "|kind=" << (binding.host_data != nullptr ? "host" : "device")
            << "|identity=" << binding.identity << "|generation=" << binding.generation
            << "|capacity=" << binding.capacity << "|offset=" << binding.offset << "|length=" << binding.length
            << "|weight=" << (binding.weight ? 1 : 0);
    }
    return { out.str() };
}

}  // namespace ggml::hrx
