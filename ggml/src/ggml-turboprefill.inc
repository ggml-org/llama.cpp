// TurboPrefill scheduler helpers.
// Included from ggml-backend.cpp after ggml_backend_sched is fully defined.

static std::vector<uint8_t> & ggml_backend_sched_tp_ram_slot(ggml_backend_sched_t sched, int slot_id, size_t size) {
    GGML_ASSERT(sched);
    GGML_ASSERT(slot_id >= 0);

    if (slot_id >= sched->tp_ram_slots_count) {
        const int new_count = slot_id + 1;
        auto * new_slots = new std::vector<uint8_t>[new_count];

        for (int i = 0; i < sched->tp_ram_slots_count; ++i) {
            new_slots[i].swap(sched->tp_ram_slots[i]);
        }

        delete[] sched->tp_ram_slots;
        sched->tp_ram_slots = new_slots;
        sched->tp_ram_slots_count = new_count;
    }

    auto & slot = sched->tp_ram_slots[slot_id];
    if (slot.size() < size) {
        slot.resize(size);
    }

    return slot;
}

static void ggml_backend_sched_tp_tensor_get_to_ram(
        ggml_backend_t backend,
        ggml_backend_event_t event,
        const struct ggml_tensor * tensor,
        void * data,
        size_t nbytes) {
    ggml_backend_tensor_get_async(backend, tensor, data, 0, nbytes);

    if (event != nullptr &&
        backend->iface.event_record != nullptr &&
        event->device->iface.event_synchronize != nullptr) {
        ggml_backend_event_record(event, backend);
        ggml_backend_event_synchronize(event);
    } else {
        ggml_backend_synchronize(backend);
    }
}

static void ggml_backend_sched_tp_tensor_get_to_ram_async(
        ggml_backend_t backend,
        const struct ggml_tensor * tensor,
        void * data,
        size_t nbytes) {
    ggml_backend_tensor_get_async(backend, tensor, data, 0, nbytes);
}

static void ggml_backend_sched_tp_tensor_set_from_ram(
        ggml_backend_t backend,
        struct ggml_tensor * tensor,
        const void * data,
        size_t nbytes) {
    ggml_backend_tensor_set_async(backend, tensor, data, 0, nbytes);
}

// Fallback copy through host RAM for inputs that cannot be copied peer-to-peer.
static void ggml_backend_sched_tp_copy_via_ram(
        ggml_backend_sched_t sched,
        int slot_id,
        ggml_backend_t input_backend,
        ggml_backend_event_t input_event,
        ggml_backend_t split_backend,
        const struct ggml_tensor * input,
        struct ggml_tensor * input_cpy) {
    GGML_ASSERT(ggml_are_same_layout(input, input_cpy) && "cannot copy tensors with different layouts");

    if (input == input_cpy) {
        return;
    }

    const size_t nbytes = ggml_nbytes(input);

    const bool has_event_wait =
            input_event != nullptr &&
            input_backend != nullptr &&
            input_backend->iface.event_record != nullptr &&
            input_event->device != nullptr &&
            input_event->device->iface.event_synchronize != nullptr;
    auto & ram_slot = ggml_backend_sched_tp_ram_slot(sched, slot_id, nbytes);
    ggml_backend_tensor_get_async(input_backend, input, ram_slot.data(), 0, nbytes);
    if (has_event_wait) {
        ggml_backend_event_record(input_event, input_backend);
        ggml_backend_event_synchronize(input_event);
    } else {
        ggml_backend_synchronize(input_backend);
    }
    ggml_backend_tensor_set_async(split_backend, input_cpy, ram_slot.data(), 0, nbytes);
}

static int ggml_backend_sched_tp_slot_id(int split_id, int input_id) {
    // Only the main activation input is pipelined through the diagonal replay.
    // One staging slot per split is enough; side inputs are restored separately.
    GGML_UNUSED(input_id);
    return split_id;
}

static std::vector<ggml_backend_sched_tp_saved_ub> & ggml_backend_sched_tp_saved_ubs(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    if (sched->tp_saved_ubs == nullptr) {
        sched->tp_saved_ubs = new std::vector<ggml_backend_sched_tp_saved_ub>();
    }
    return *sched->tp_saved_ubs;
}

static void ggml_backend_sched_tp_capture_tensor_meta(
        std::vector<ggml_backend_sched_tp_tensor_meta> & metas,
        struct ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return;
    }

    for (const auto & meta : metas) {
        if (meta.tensor == tensor) {
            return;
        }
    }

    ggml_backend_sched_tp_tensor_meta meta;
    meta.tensor    = tensor;
    meta.type      = tensor->type;
    meta.data      = tensor->data;
    meta.buffer    = tensor->buffer;
    meta.view_src  = tensor->view_src;
    meta.view_offs = tensor->view_offs;
    meta.flags     = tensor->flags;

    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        meta.ne[i] = tensor->ne[i];
        meta.nb[i] = tensor->nb[i];
    }

    metas.push_back(meta);
}

static void ggml_backend_sched_tp_apply_tensor_metas(
        const std::vector<ggml_backend_sched_tp_tensor_meta> & metas) {
    for (const auto & meta : metas) {
        struct ggml_tensor * tensor = meta.tensor;
        if (tensor == nullptr) {
            continue;
        }

        tensor->type      = meta.type;
        tensor->data      = meta.data;
        tensor->buffer    = meta.buffer;
        tensor->view_src  = meta.view_src;
        tensor->view_offs = meta.view_offs;
        tensor->flags     = meta.flags;

        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            tensor->ne[i] = meta.ne[i];
            tensor->nb[i] = meta.nb[i];
        }
    }
}

static void ggml_backend_sched_tp_store_to_ram_slot_async(
        ggml_backend_sched_t sched,
        int slot_id,
        ggml_backend_t src_backend,
        const struct ggml_tensor * src) {
    const size_t nbytes = ggml_nbytes(src);
    auto & ram_slot = ggml_backend_sched_tp_ram_slot(sched, slot_id, nbytes);

    ggml_backend_sched_tp_tensor_get_to_ram_async(src_backend, src, ram_slot.data(), nbytes);
}

static bool ggml_backend_sched_tp_load_from_ram_slot(
        ggml_backend_sched_t sched,
        int slot_id,
        ggml_backend_t dst_backend,
        struct ggml_tensor * dst) {
    const size_t nbytes = ggml_nbytes(dst);
    auto & ram_slot = ggml_backend_sched_tp_ram_slot(sched, slot_id, 0);

    if (ram_slot.size() < nbytes) {
        GGML_LOG_ERROR("%s: staging slot %d is not ready: have %zu bytes, need %zu bytes\n",
                __func__, slot_id, ram_slot.size(), nbytes);
        return false;
    }

    ggml_backend_sched_tp_tensor_set_from_ram(dst_backend, dst, ram_slot.data(), nbytes);

    return true;
}
