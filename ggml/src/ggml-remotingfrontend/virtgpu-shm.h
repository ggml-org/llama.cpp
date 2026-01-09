#pragma once

#include <cassert>
#include <cstdint>
#include <cstddef>
#include <atomic>
#include <sys/mman.h>

#include "virtgpu-utils.h"

struct virtgpu;

struct virtgpu_shmem {
    uint32_t res_id;
    size_t mmap_size;
    void *mmap_ptr;

    uint32_t gem_handle;
};

int virtgpu_shmem_create(struct virtgpu *gpu, size_t size, struct virtgpu_shmem *shmem);
void virtgpu_shmem_destroy(struct virtgpu *gpu, struct virtgpu_shmem *shmem);
