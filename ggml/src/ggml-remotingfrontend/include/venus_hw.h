/*
 * Copyright 2020 Chromium
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef VENUS_HW_H
#define VENUS_HW_H

#include <stdint.h>

struct virgl_renderer_capset_venus {
   uint32_t wire_format_version;
   uint32_t vk_xml_version;
   uint32_t vk_ext_command_serialization_spec_version;
   uint32_t vk_mesa_venus_protocol_spec_version;

   /* This flag indicates render server config, and will be needed until drm
    * virtio-gpu blob mem gets fixed to attach_resource before resource_map.
    */
   uint32_t supports_blob_id_0;

   /* Extension number N, where N is defined by the Vulkan spec, corresponds
    * to bit [N / 32] & (1 << N % 32). The below mask1 covers the first 1023
    * Vulkan extensions (numbered from 1 to 1023).
    *
    * Bit (mask1[0] & 0x1) is used for backward compatibility purpose. When
    * that bit is set, the extension mask(s) are valid. Otherwise, all the
    * extensions are assumed to be supported by the renderer side protocol.
    */
   uint32_t vk_extension_mask1[32];

   /* The single-threaded renderer cannot afford potential blocking calls. It
    * also leads to GPU lost if the wait depends on a following command. This
    * capset allows such blocking calls to passthrough from the clients, and
    * shifts the responsibilities to the client drivers.
    */
   uint32_t allow_vk_wait_syncs;

   /* This flag indicates that the renderer supports multiple fencing
    * timelines. The client driver is expected to associate each VkQueue with
    * one of these timelines at queue creation by binding it with an unused
    * ring_idx. Queues created without a ring_idx binding are associated to a
    * shared legacy timeline. The special ring_idx==0 is reserved for CPU
    * fences that are signaled by the renderer immediately upon consumption of
    * the associated renderer submission.
    */
   uint32_t supports_multiple_timelines;

   /* This flag indicates to the guest that hypervisor does not support memory
    * pages injections and blob allocations must be done by guest from the
    * dedicated heap (Host visible memory).
    */
   uint32_t use_guest_vram;
};

#endif /* VENUS_HW_H */
