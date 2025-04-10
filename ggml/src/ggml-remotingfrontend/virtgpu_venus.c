static inline void vn_encode_vkEnumeratePhysicalDevices(struct vn_cs_encoder *enc, VkCommandFlagsEXT cmd_flags, VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices)
{
    const VkCommandTypeEXT cmd_type = VK_COMMAND_TYPE_vkEnumeratePhysicalDevices_EXT;

    vn_encode_VkCommandTypeEXT(enc, &cmd_type);
    vn_encode_VkFlags(enc, &cmd_flags);

    vn_encode_VkInstance(enc, &instance);
    if (vn_encode_simple_pointer(enc, pPhysicalDeviceCount))
        vn_encode_uint32_t(enc, pPhysicalDeviceCount);
    if (pPhysicalDevices) {
        vn_encode_array_size(enc, (pPhysicalDeviceCount ? *pPhysicalDeviceCount : 0));
        for (uint32_t i = 0; i < (pPhysicalDeviceCount ? *pPhysicalDeviceCount : 0); i++)
            vn_encode_VkPhysicalDevice(enc, &pPhysicalDevices[i]);
    } else {
        vn_encode_array_size(enc, 0);
    }
}

static inline struct vn_cs_encoder *
vn_ring_submit_command_init(struct vn_ring *ring,
                            struct vn_ring_submit_command *submit,
                            void *cmd_data,
                            size_t cmd_size,
                            size_t reply_size)
{
   submit->buffer = VN_CS_ENCODER_BUFFER_INITIALIZER(cmd_data);
   submit->command = VN_CS_ENCODER_INITIALIZER(&submit->buffer, cmd_size);

   submit->reply_size = reply_size;
   submit->reply_shmem = NULL;

   submit->ring_seqno_valid = false;

   return &submit->command;
}

static inline void vn_submit_vkEnumeratePhysicalDevices(struct vn_ring *vn_ring, VkCommandFlagsEXT cmd_flags, VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices, struct vn_ring_submit_command *submit)
{
    uint8_t local_cmd_data[VN_SUBMIT_LOCAL_CMD_SIZE];
    void *cmd_data = local_cmd_data;
    size_t cmd_size = vn_sizeof_vkEnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    if (cmd_size > sizeof(local_cmd_data)) {
        cmd_data = malloc(cmd_size);
        if (!cmd_data)
            cmd_size = 0;
    }
    const size_t reply_size = cmd_flags & VK_COMMAND_GENERATE_REPLY_BIT_EXT ? vn_sizeof_vkEnumeratePhysicalDevices_reply(instance, pPhysicalDeviceCount, pPhysicalDevices) : 0;

    struct vn_cs_encoder *enc = vn_ring_submit_command_init(vn_ring, submit, cmd_data, cmd_size, reply_size);
    if (cmd_size) {
        vn_encode_vkEnumeratePhysicalDevices(enc, cmd_flags, instance, pPhysicalDeviceCount, pPhysicalDevices);
        vn_ring_submit_command(vn_ring, submit);
        if (cmd_data != local_cmd_data)
            free(cmd_data);
    }
}

VkResult vn_call_vkEnumeratePhysicalDevices(struct vn_ring *vn_ring, VkInstance instance, uint32_t* pPhysicalDeviceCount, VkPhysicalDevice* pPhysicalDevices)
{
    VN_TRACE_FUNC();

    struct vn_ring_submit_command submit;
    vn_submit_vkEnumeratePhysicalDevices(vn_ring, VK_COMMAND_GENERATE_REPLY_BIT_EXT, instance, pPhysicalDeviceCount, pPhysicalDevices, &submit);
    struct vn_cs_decoder *dec = vn_ring_get_command_reply(vn_ring, &submit);
    if (dec) {
        const VkResult ret = vn_decode_vkEnumeratePhysicalDevices_reply(dec, instance, pPhysicalDeviceCount, pPhysicalDevices);
        vn_ring_free_command_reply(vn_ring, &submit);
        return ret;
    } else {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
}

VkResult
vn_ring_submit_command_simple(struct vn_ring *ring,
                              const struct vn_cs_encoder *cs)
{
   mtx_lock(&ring->mutex);
   VkResult result = vn_ring_submit_locked(ring, cs, NULL, NULL);
   mtx_unlock(&ring->mutex);

   return result;
}

static VkResult
vn_ring_submit_locked(struct vn_ring *ring,
                      const struct vn_cs_encoder *cs,
                      struct vn_renderer_shmem *extra_shmem,
                      uint32_t *ring_seqno)
{
   const bool direct = vn_ring_submission_can_direct(ring, cs);
   if (!direct && cs->storage_type == VN_CS_ENCODER_STORAGE_POINTER) {
      cs = vn_ring_cs_upload_locked(ring, cs);
      if (!cs)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      assert(cs->storage_type != VN_CS_ENCODER_STORAGE_POINTER);
   }

   struct vn_ring_submission submit;
   VkResult result =
      vn_ring_submission_prepare(ring, &submit, cs, extra_shmem, direct);
   if (result != VK_SUCCESS)
      return result;

   uint32_t seqno;
   const bool notify =
      vn_ring_submit_internal(ring, submit.submit, submit.cs, &seqno);
   if (notify) {
      uint32_t notify_ring_data[8];
      struct vn_cs_encoder local_enc = VN_CS_ENCODER_INITIALIZER_LOCAL(
         notify_ring_data, sizeof(notify_ring_data));
      vn_encode_vkNotifyRingMESA(&local_enc, 0, ring->id, seqno, 0);
      vn_renderer_submit_simple(ring->instance->renderer, notify_ring_data,
                                vn_cs_encoder_get_len(&local_enc));
   }

   vn_ring_submission_cleanup(&submit);

   if (ring_seqno)
      *ring_seqno = seqno;

   return VK_SUCCESS;
}

static VkResult
vn_ring_submission_prepare(struct vn_ring *ring,
                           struct vn_ring_submission *submit,
                           const struct vn_cs_encoder *cs,
                           struct vn_renderer_shmem *extra_shmem,
                           bool direct)
{
   submit->cs = vn_ring_submission_get_cs(submit, cs, direct);
   if (!submit->cs)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   submit->submit =
      vn_ring_submission_get_ring_submit(ring, cs, extra_shmem, direct);
   if (!submit->submit) {
      vn_ring_submission_cleanup(submit);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }

   return VK_SUCCESS;
}

static bool
vn_ring_submit_internal(struct vn_ring *ring,
                        struct vn_ring_submit *submit,
                        const struct vn_cs_encoder *cs,
                        uint32_t *seqno)
{
   /* write cs to the ring */
   assert(!vn_cs_encoder_is_empty(cs));

   /* avoid -Wmaybe-unitialized */
   uint32_t cur_seqno = 0;

   for (uint32_t i = 0; i < cs->buffer_count; i++) {
      const struct vn_cs_encoder_buffer *buf = &cs->buffers[i];
      cur_seqno = vn_ring_wait_space(ring, buf->committed_size);
      vn_ring_write_buffer(ring, buf->base, buf->committed_size);
   }

   vn_ring_store_tail(ring);
   const VkRingStatusFlagsMESA status = vn_ring_load_status(ring);
   if (status & VK_RING_STATUS_FATAL_BIT_MESA) {
      vn_log(NULL, "vn_ring_submit abort on fatal");
      abort();
   }

   vn_ring_retire_submits(ring, cur_seqno);

   submit->seqno = ring->cur;
   list_addtail(&submit->head, &ring->submits);

   *seqno = submit->seqno;

   /* Notify renderer to wake up idle ring if at least VN_RING_IDLE_TIMEOUT_NS
    * has passed since the last sent notification to avoid excessive wake up
    * calls (non-trivial since submitted via virtio-gpu kernel).
    */
   if (status & VK_RING_STATUS_IDLE_BIT_MESA) {
      const int64_t now = os_time_get_nano();
      if (os_time_timeout(ring->last_notify, ring->next_notify, now)) {
         ring->last_notify = now;
         ring->next_notify = now + VN_RING_IDLE_TIMEOUT_NS;
         return true;
      }
   }
   return false;
}

static void
vn_ring_write_buffer(struct vn_ring *ring, const void *data, uint32_t size)
{
   assert(ring->cur + size - vn_ring_load_head(ring) <= ring->buffer_size);

   const uint32_t offset = ring->cur & ring->buffer_mask;
   if (offset + size <= ring->buffer_size) {
      memcpy(ring->shared.buffer + offset, data, size);
   } else {
      const uint32_t s = ring->buffer_size - offset;
      memcpy(ring->shared.buffer + offset, data, s);
      memcpy(ring->shared.buffer, data + s, size - s);
   }

   ring->cur += size;
}
