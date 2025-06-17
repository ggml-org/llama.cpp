#pragma once

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <sys/mman.h>

#include "virtgpu.h"
#include "virtgpu-utils.h"

struct vn_refcount {
   int count; //atomic_int
};


struct vn_renderer_shmem {
   struct vn_refcount refcount;

   uint32_t res_id;
   size_t mmap_size; /* for internal use only (i.e., munmap) */
   void *mmap_ptr;

   struct list_head cache_head;
   int64_t cache_timestamp;

   uint32_t gem_handle;

   struct virtgpu_shmem *shmem;
};

struct vn_renderer_shmem *virtgpu_shmem_create(struct virtgpu *gpu, size_t size);
void virtgpu_shmem_destroy(struct virtgpu *gpu, struct virtgpu_shmem *shmem);


struct virtgpu_shmem {
   struct vn_renderer_shmem base;
   uint32_t gem_handle;
};
