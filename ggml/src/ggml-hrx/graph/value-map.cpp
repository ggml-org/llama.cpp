#include "value-map.h"

#include "ggml-impl.h"

#include <utility>

namespace ggml::hrx {
namespace {

static const ggml_tensor * tensor_storage_root(const ggml_tensor * tensor) {
    while (tensor != nullptr && tensor->view_src != nullptr) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static size_t tensor_storage_offset(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->view_src != nullptr ? tensor->view_offs : 0;
}

static bool tensor_storage_relative_offset(const ggml_tensor * source, const ggml_tensor * tensor, size_t & offset) {
    if (source == nullptr || tensor == nullptr || tensor_storage_root(source) != tensor_storage_root(tensor)) {
        return false;
    }
    const size_t source_offset = tensor_storage_offset(source);
    const size_t tensor_offset = tensor_storage_offset(tensor);
    if (tensor_offset < source_offset) {
        return false;
    }
    const size_t relative_offset = tensor_offset - source_offset;
    if (relative_offset > ggml_nbytes(source) || ggml_nbytes(tensor) > ggml_nbytes(source) - relative_offset) {
        return false;
    }
    offset = relative_offset;
    return true;
}

}  // namespace

const Value * ValueMap::find_alias_source(const ggml_tensor * tensor) const {
    if (tensor == nullptr || tensor->view_src == nullptr) {
        return nullptr;
    }
    const Value * exact_source = find_tensor(tensor->view_src);
    if (exact_source != nullptr) {
        return exact_source;
    }
    const Value * best_source      = nullptr;
    size_t        best_source_size = 0;
    for (const Value & value : values_) {
        size_t relative_offset = 0;
        if (value.tensor == nullptr || !tensor_storage_relative_offset(value.tensor, tensor, relative_offset)) {
            continue;
        }
        if (best_source == nullptr || ggml_nbytes(value.tensor) < best_source_size) {
            best_source      = &value;
            best_source_size = ggml_nbytes(value.tensor);
        }
    }
    return best_source;
}

void ValueMap::promote_storage_root_external(ValueId storage_root) {
    if (storage_root.value < 0 || static_cast<size_t>(storage_root.value) >= values_.size()) {
        return;
    }
    values_[static_cast<size_t>(storage_root.value)].kind = ValueKind::External;
}

ValueId ValueMap::get_or_add_tensor_value(const ggml_tensor * tensor, ValueKind kind) {
    const auto found = tensor_values_.find(tensor);
    if (found != tensor_values_.end()) {
        Value & value = values_[found->second];
        if (kind == ValueKind::External) {
            value.kind = ValueKind::External;
            promote_storage_root_external(value.storage_root);
        }
        return value.id;
    }

    const ValueId  id(static_cast<int32_t>(values_.size()));
    const Value *  alias_source = find_alias_source(tensor);
    ValueStorageId storage;
    ValueId        storage_root;
    ValueId        alias_source_id;
    size_t         storage_offset     = 0;
    size_t         storage_byte_count = ggml_nbytes(tensor);
    if (alias_source != nullptr) {
        size_t relative_offset = 0;
        if (!tensor_storage_relative_offset(alias_source->tensor, tensor, relative_offset)) {
            relative_offset = tensor->view_offs;
        }
        storage            = alias_source->storage;
        storage_root       = alias_source->storage_root;
        alias_source_id    = alias_source->id;
        storage_offset     = alias_source->storage_offset + relative_offset;
        storage_byte_count = alias_source->storage_byte_count;
        const Value * root = find(storage_root);
        if (root != nullptr && root->kind == ValueKind::External) {
            kind = ValueKind::External;
        }
        if (kind == ValueKind::External) {
            promote_storage_root_external(storage_root);
        }
    } else {
        storage      = ValueStorageId(static_cast<int32_t>(storages_.size()));
        storage_root = id;
        storages_.push_back({ storage, storage_root, storage_byte_count });
    }

    Value value = {
        id,
        kind,
        storage,
        storage_root,
        alias_source_id,
        storage_offset,
        storage_byte_count,
        tensor->type,
        {},
        {},
        ggml_nelements(tensor),
        ggml_nbytes(tensor),
        ggml_is_contiguous(tensor),
        tensor,
        std::nullopt,
    };
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        value.ne[i] = tensor->ne[i];
        value.nb[i] = tensor->nb[i];
    }

    values_.push_back(std::move(value));
    tensor_values_.emplace(tensor, values_.size() - 1);
    return values_.back().id;
}

const Value * ValueMap::find(ValueId id) const {
    if (id.value < 0 || static_cast<size_t>(id.value) >= values_.size()) {
        return nullptr;
    }
    return &values_[static_cast<size_t>(id.value)];
}

const ValueStorage * ValueMap::find_storage(ValueStorageId id) const {
    if (id.value < 0 || static_cast<size_t>(id.value) >= storages_.size()) {
        return nullptr;
    }
    return &storages_[static_cast<size_t>(id.value)];
}

bool ValueMap::bind_buffer(ValueId id, ValueBufferBinding binding) {
    if (id.value < 0 || static_cast<size_t>(id.value) >= values_.size()) {
        return false;
    }
    Value & value = values_[static_cast<size_t>(id.value)];
    if (value.kind != ValueKind::External) {
        return false;
    }
    value.buffer = std::move(binding);
    return true;
}

std::optional<ValueBufferBinding> ValueMap::resolve_buffer_binding(ValueId id) const {
    const Value * value = find(id);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (value->buffer.has_value()) {
        return value->buffer;
    }
    if (value->storage_root == value->id) {
        return std::nullopt;
    }
    const Value * root = find(value->storage_root);
    if (root == nullptr || !root->buffer.has_value()) {
        return std::nullopt;
    }
    ValueBufferBinding binding = *root->buffer;
    if (value->storage_offset > binding.length) {
        return std::nullopt;
    }
    if (value->byte_count > binding.length - value->storage_offset) {
        return std::nullopt;
    }
    binding.offset += value->storage_offset;
    binding.length = value->byte_count;
    return binding;
}

std::vector<ValueId> ValueMap::external_value_ids() const {
    std::vector<ValueId> ids;
    for (const Value & value : values_) {
        if (value.kind == ValueKind::External) {
            ids.push_back(value.id);
        }
    }
    return ids;
}

Status ValueMap::alias_storage(ValueId target, ValueId source) {
    Status status;
    if (target.value < 0 || static_cast<size_t>(target.value) >= values_.size()) {
        status.log("value alias target %d does not exist", target.value);
        return status;
    }
    if (source.value < 0 || static_cast<size_t>(source.value) >= values_.size()) {
        status.log("value alias source %d does not exist", source.value);
        return status;
    }
    if (target == source) {
        status.log("value alias target %d aliases itself", target.value);
        return status;
    }

    Value &       target_value = values_[static_cast<size_t>(target.value)];
    const Value & source_value = values_[static_cast<size_t>(source.value)];
    if (target_value.storage == source_value.storage) {
        return status;
    }
    if (target_value.kind != ValueKind::Transient) {
        status.log("value alias target %d is not transient", target.value);
        return status;
    }
    if (target_value.type != source_value.type || target_value.byte_count != source_value.byte_count ||
        target_value.element_count != source_value.element_count ||
        target_value.contiguous != source_value.contiguous) {
        status.log("value alias target %d is incompatible with source %d", target.value, source.value);
        return status;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (target_value.ne[i] != source_value.ne[i] || target_value.nb[i] != source_value.nb[i]) {
            status.log("value alias target %d has a different layout than source %d", target.value, source.value);
            return status;
        }
    }
    if (target_value.alias_source.value >= 0 && target_value.alias_source != source) {
        status.log("value alias target %d already aliases source %d", target.value, target_value.alias_source.value);
        return status;
    }

    target_value.storage            = source_value.storage;
    target_value.storage_root       = source_value.storage_root;
    target_value.alias_source       = source;
    target_value.storage_offset     = source_value.storage_offset;
    target_value.storage_byte_count = source_value.storage_byte_count;
    return status;
}

ValueId ValueMap::storage_root(ValueId id) const {
    const Value * value = find(id);
    return value == nullptr ? ValueId() : value->storage_root;
}

bool ValueMap::same_storage(ValueId lhs, ValueId rhs) const {
    const Value * lhs_value = find(lhs);
    const Value * rhs_value = find(rhs);
    return lhs_value != nullptr && rhs_value != nullptr && lhs_value->storage == rhs_value->storage;
}

Status ValueMap::add_snapshot_storage(ValueStorage storage) {
    Status status;
    if (storage.id.value < 0 || static_cast<size_t>(storage.id.value) != storages_.size()) {
        status.log("snapshot storage id %d is not the next storage id %zu", storage.id.value, storages_.size());
        return status;
    }
    if (storage.root.value < 0) {
        status.log("snapshot storage %d has invalid root value %d", storage.id.value, storage.root.value);
        return status;
    }
    storages_.push_back(storage);
    return status;
}

Status ValueMap::add_snapshot_value(Value value) {
    Status status;
    if (value.id.value < 0 || static_cast<size_t>(value.id.value) != values_.size()) {
        status.log("snapshot value id %d is not the next value id %zu", value.id.value, values_.size());
        return status;
    }
    if (value.storage.value < 0 || static_cast<size_t>(value.storage.value) >= storages_.size()) {
        status.log("snapshot value %d references missing storage %d", value.id.value, value.storage.value);
        return status;
    }
    if (value.storage_root.value < 0 || static_cast<size_t>(value.storage_root.value) > values_.size()) {
        status.log("snapshot value %d references invalid storage root %d", value.id.value, value.storage_root.value);
        return status;
    }
    if (value.alias_source.value >= 0 && static_cast<size_t>(value.alias_source.value) >= values_.size()) {
        status.log("snapshot value %d references missing alias source %d", value.id.value, value.alias_source.value);
        return status;
    }
    value.tensor = nullptr;
    value.buffer.reset();
    values_.push_back(std::move(value));
    return status;
}

const Value * ValueMap::find_tensor(const ggml_tensor * tensor) const {
    const auto found = tensor_values_.find(tensor);
    if (found == tensor_values_.end()) {
        return nullptr;
    }
    return &values_[found->second];
}

}  // namespace ggml::hrx
