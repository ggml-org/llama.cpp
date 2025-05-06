#pragma once

#include <xf86drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdbool.h>
#include <threads.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "virtgpu-utils.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/api_remoting.h"
#include "/Users/kevinpouget/remoting/llama_cpp/src/ggml/src/ggml-remotingbackend/shared/venus_cs.h"

void thks_bye();

#include "virtgpu-shm.h"

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#include "drm-uapi/virtgpu_drm.h"
#include "virglrenderer_hw.h"
#include "venus_hw.h"

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID 0x1af4
#define VIRTGPU_PCI_DEVICE_ID 0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM 9

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

#define VN_DEBUG(what) true

typedef enum virt_gpu_result_t {
    APIR_SUCCESS = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
} virt_gpu_result_t;


struct remoting_dev_instance {
  int yes;
};

#define PRINTFLIKE(f, a) __attribute__ ((format(__printf__, f, a)))

inline void
vn_log(struct remoting_dev_instance *instance, const char *format, ...)
   PRINTFLIKE(2, 3);


struct virtgpu {
   struct remoting_dev_instance *instance;

   int fd;

   bool has_primary;
   int primary_major;
   int primary_minor;
   int render_major;
   int render_minor;

   int bustype;
   drmPciBusInfo pci_bus_info;

   uint32_t max_timeline_count;

   struct {
      enum virgl_renderer_capset id;
      uint32_t version;
      struct virgl_renderer_capset_venus data;
   } capset;

   uint32_t shmem_blob_mem;
   uint32_t bo_blob_mem;

   /* note that we use gem_handle instead of res_id to index because
    * res_id is monotonically increasing by default (see
    * virtio_gpu_resource_id_get)
    */
  struct util_sparse_array shmem_array;
  // struct util_sparse_array bo_array;

   mtx_t dma_buf_import_mutex;

  //   struct virtgpu_shmem_cache shmem_cache;

   bool supports_cross_device;

  /* KP */
  struct vn_renderer_shmem *reply_shmem;
};


static inline int
virtgpu_ioctl(struct virtgpu *gpu, unsigned long request, void *args)
{
   return drmIoctl(gpu->fd, request, args);
}

void create_virtgpu();
static virt_gpu_result_t virtgpu_open_device(struct virtgpu *gpu, const drmDevicePtr dev);
static virt_gpu_result_t virtgpu_open(struct virtgpu *gpu);


static virt_gpu_result_t virtgpu_init_params(struct virtgpu *gpu);
static virt_gpu_result_t virtgpu_init_capset(struct virtgpu *gpu);
static virt_gpu_result_t virtgpu_init_context(struct virtgpu *gpu);

static int virtgpu_ioctl_context_init(struct virtgpu *gpu,
				      enum virgl_renderer_capset capset_id);
static int
virtgpu_ioctl_get_caps(struct virtgpu *gpu,
                       enum virgl_renderer_capset id,
                       uint32_t version,
                       void *capset,
                       size_t capset_size);
static uint64_t virtgpu_ioctl_getparam(struct virtgpu *gpu, uint64_t param);
static void virtgpu_init_renderer_info(struct virtgpu *gpu);
static int remote_call(struct virtgpu *gpu, int32_t cmd_type, int32_t cmd_flags, int32_t arg1, int32_t arg2, int32_t arg3);
