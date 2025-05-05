#pragma once

#include <xf86drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdbool.h>
#include <threads.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/sysmacros.h>

void breakpoint();

#include "virtgpu-shm.h"
#include "virtgpu-utils.h"

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

typedef enum VkResult {
    VK_SUCCESS = 0,
    VK_NOT_READY = 1,
    VK_TIMEOUT = 2,
    VK_EVENT_SET = 3,
    VK_EVENT_RESET = 4,
    VK_INCOMPLETE = 5,
    VK_ERROR_OUT_OF_HOST_MEMORY = -1,
    VK_ERROR_OUT_OF_DEVICE_MEMORY = -2,
    VK_ERROR_INITIALIZATION_FAILED = -3,
    VK_ERROR_DEVICE_LOST = -4,
    VK_ERROR_MEMORY_MAP_FAILED = -5,
    VK_ERROR_LAYER_NOT_PRESENT = -6,
    VK_ERROR_EXTENSION_NOT_PRESENT = -7,
    VK_ERROR_FEATURE_NOT_PRESENT = -8,
    VK_ERROR_INCOMPATIBLE_DRIVER = -9,
    VK_ERROR_TOO_MANY_OBJECTS = -10,
    VK_ERROR_FORMAT_NOT_SUPPORTED = -11,
    VK_ERROR_FRAGMENTED_POOL = -12,
    VK_ERROR_UNKNOWN = -13,
    VK_ERROR_OUT_OF_POOL_MEMORY = -1000069000,
    VK_ERROR_INVALID_EXTERNAL_HANDLE = -1000072003,
    VK_ERROR_FRAGMENTATION = -1000161000,
    VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS = -1000257000,
    VK_PIPELINE_COMPILE_REQUIRED = 1000297000,
    VK_ERROR_SURFACE_LOST_KHR = -1000000000,
    VK_ERROR_NATIVE_WINDOW_IN_USE_KHR = -1000000001,
    VK_SUBOPTIMAL_KHR = 1000001003,
    VK_ERROR_OUT_OF_DATE_KHR = -1000001004,
    VK_ERROR_INCOMPATIBLE_DISPLAY_KHR = -1000003001,
    VK_ERROR_VALIDATION_FAILED_EXT = -1000011001,
    VK_ERROR_INVALID_SHADER_NV = -1000012000,
    VK_ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR = -1000023000,
    VK_ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR = -1000023001,
    VK_ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR = -1000023002,
    VK_ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR = -1000023003,
    VK_ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR = -1000023004,
    VK_ERROR_VIDEO_STD_VERSION_NOT_SUPPORTED_KHR = -1000023005,
    VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT = -1000158000,
    VK_ERROR_NOT_PERMITTED_KHR = -1000174001,
    VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT = -1000255000,
    VK_THREAD_IDLE_KHR = 1000268000,
    VK_THREAD_DONE_KHR = 1000268001,
    VK_OPERATION_DEFERRED_KHR = 1000268002,
    VK_OPERATION_NOT_DEFERRED_KHR = 1000268003,
    VK_ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR = -1000299000,
    VK_ERROR_COMPRESSION_EXHAUSTED_EXT = -1000338000,
    VK_INCOMPATIBLE_SHADER_BINARY_EXT = 1000482000,
    VK_ERROR_OUT_OF_POOL_MEMORY_KHR = VK_ERROR_OUT_OF_POOL_MEMORY,
    VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR = VK_ERROR_INVALID_EXTERNAL_HANDLE,
    VK_ERROR_FRAGMENTATION_EXT = VK_ERROR_FRAGMENTATION,
    VK_ERROR_NOT_PERMITTED_EXT = VK_ERROR_NOT_PERMITTED_KHR,
    VK_ERROR_INVALID_DEVICE_ADDRESS_EXT = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
    VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS_KHR = VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS,
    VK_PIPELINE_COMPILE_REQUIRED_EXT = VK_PIPELINE_COMPILE_REQUIRED,
    VK_ERROR_PIPELINE_COMPILE_REQUIRED_EXT = VK_PIPELINE_COMPILE_REQUIRED,
    VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT = VK_INCOMPATIBLE_SHADER_BINARY_EXT,
    VK_RESULT_MAX_ENUM = 0x7FFFFFFF
} VkResult;


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
};


static inline int
virtgpu_ioctl(struct virtgpu *gpu, unsigned long request, void *args)
{
   return drmIoctl(gpu->fd, request, args);
}

void create_virtgpu();
static VkResult virtgpu_open_device(struct virtgpu *gpu, const drmDevicePtr dev);
static VkResult virtgpu_open(struct virtgpu *gpu);


static VkResult virtgpu_init_params(struct virtgpu *gpu);
static VkResult virtgpu_init_capset(struct virtgpu *gpu);
static VkResult virtgpu_init_context(struct virtgpu *gpu);

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
static int virtgpu_submit(struct virtgpu *gpu, struct vn_renderer_shmem *shmem);
