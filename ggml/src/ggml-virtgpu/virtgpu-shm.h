#pragma once

#include "virtgpu-utils.h"

#include <sys/mman.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>

struct virtgpu;

struct virtgpu_shmem {
    uint32_t res_id;
    size_t   mmap_size;
    void *   mmap_ptr;

    uint32_t gem_handle;
};

int  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);
void virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);
