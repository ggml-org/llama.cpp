#include "ggml-et.h"
#include "ggml-et-common.h"
#include "ggml-et-kernels.h"
#include "ggml-et-memops.h"
#include "ggml-et-ops.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"
#include <stdarg.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "cannot include the filesystem library"
#endif

/*
  ET Driver.

  `ggml_et_driver()` handles both the device layer and the runtime,
  for doing actual operations on devices.
*/


/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
static void ggml_et_dump_tensor_metadata(const ggml_tensor* ggtensor, size_t indent_level, const char* title)
{
    char* spaces = (char*)alloca(indent_level+1);
    memset(spaces, ' ', indent_level);
    spaces[indent_level] = '\0';
    fprintf(stderr, "%s%s: %s\n"
        "%s  type: %s\n"
        "%s  ne: %lld %lld %lld %lld\n"
        "%s  nb: %zu %zu %zu %zu\n"
        "%s  op: %s\n"
        "%s  data: %p\n"
        "%s  src0: %p\n",
        spaces, title, ggtensor->name,
        spaces, ggml_type_name(ggtensor->type),
        spaces, (long long)ggtensor->ne[0], (long long)ggtensor->ne[1], (long long)ggtensor->ne[2], (long long)ggtensor->ne[3],
        spaces, ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3],
        spaces, ggml_op_name(ggtensor->op),
        spaces, ggtensor->data,
        spaces, (void *)ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
static void ggml_et_dump_operator_metadata(const ggml_tensor* ggtensor)
{
    GGML_ASSERT(ggtensor != NULL);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    for(int i=0;i<GGML_MAX_SRC && ggtensor->src[i];i++) {
        char arr[16];
        int n = snprintf(arr, sizeof(arr), "src[%i]->name", i);
        GGML_ASSERT((unsigned)n < sizeof(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    if(ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

static struct ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer> device_layer;
    std::shared_ptr<rt::IRuntime> runtime;
    std::unique_ptr<std::ofstream> profile_stream;
    std::unique_ptr<std::ofstream> kernel_id_stream;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    bool profiling_enabled = false;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
static std::string ggml_et_get_default_et_path() {
    // List of environment variables to check in order of preference
    const char* const env_vars[] = {"ET_TOOLCHAIN", "TOOLCHAIN_ROOT"};

    for (const char* var : env_vars) {
        if (const char* et_path = std::getenv(var)) {
            if (et_path && *et_path != '\0') {
                return fs::path(et_path).string();
            }
        }
    }

    // Otherwise assume default
    return fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
static inline auto ggml_et_get_default_sysemu_options() {
    constexpr uint64_t kSysEmuMaxCycles = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;
    const std::string et_path = ggml_et_get_default_et_path() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootromTrampolineToBL2ElfPath = et_path + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.spBL2ElfPath = et_path + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machineMinionElfPath = et_path + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.masterMinionElfPath = et_path + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.workerMinionElfPath = et_path + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executablePath = et_path + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    const std::vector<std::string> required_files = {
        sysEmuOptions.bootromTrampolineToBL2ElfPath,
        sysEmuOptions.spBL2ElfPath,
        sysEmuOptions.machineMinionElfPath,
        sysEmuOptions.masterMinionElfPath,
        sysEmuOptions.workerMinionElfPath,
        sysEmuOptions.executablePath,
    };

    for (const auto& file : required_files) {
        if (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.runDir = (fs::current_path().string() + "/");
    sysEmuOptions.maxCycles = kSysEmuMaxCycles;
    sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
    sysEmuOptions.puUart0Path = sysEmuOptions.runDir + "pu_uart0_tx.log";
    sysEmuOptions.puUart1Path = sysEmuOptions.runDir + "pu_uart1_tx.log";
    sysEmuOptions.spUart0Path = sysEmuOptions.runDir + "spio_uart0_tx.log";
    sysEmuOptions.spUart1Path = sysEmuOptions.runDir + "spio_uart1_tx.log";
    sysEmuOptions.startGdb = false;
    sysEmuOptions.memcheck = false;

    return sysEmuOptions;
}

// Forward declaration
static void ggml_et_driver_cleanup();

static bool ggml_et_driver_init() {
    if (_drv.runtime != nullptr) {
	assert(_drv.device_layer != nullptr);
    } else {
	try {
        #if defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
        // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
        _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
        #else
        // For physical PCIe device
        _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
        #endif

	    _drv.runtime = rt::IRuntime::create(_drv.device_layer);

	    // Initialize profiler if requested via environment variable
	    const char* profile_path = getenv("GGML_ET_PROFILE");
	    if (profile_path) {
	        std::string output_path = std::string(profile_path) + "/et_runtime_trace.json";
			std::string kernel_id_path = std::string(profile_path) + "/kernel_id.json";

	        _drv.profile_stream = std::make_unique<std::ofstream>(output_path);
			_drv.kernel_id_stream = std::make_unique<std::ofstream>(kernel_id_path);
	        if (!_drv.profile_stream->is_open()) {
	            GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_path.c_str());
				abort();
	        }
			if(!_drv.kernel_id_stream->is_open()) {
			    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_path.c_str());
				abort();
			}

            auto* profiler = _drv.runtime->getProfiler();
            profiler->start(*_drv.profile_stream, rt::IProfiler::OutputType::Json);
            _drv.profiling_enabled = true;
            GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

            // Register cleanup at program exit
            std::atexit(ggml_et_driver_cleanup);
	    }
	} catch (const std::exception& e) {
	    GGML_LOG_ERROR("ggml_et: %s", e.what());
	    if (_drv.device_layer != nullptr)
		_drv.device_layer.reset();
	    if (_drv.runtime != nullptr)
		_drv.runtime.reset();
	    return false;
	}
    }
    return true;
}

static std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    return _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    return _drv.runtime;
}

static void ggml_et_driver_cleanup() {
    if (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        auto* profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = false;

        if (_drv.profile_stream) {
            _drv.profile_stream->close();
            _drv.profile_stream.reset();
        }

        // Save kernel map
        if (_drv.kernel_id_stream && !_drv.kernel_map.empty()) {
            auto & os = *_drv.kernel_id_stream;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            for (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].first << "\": " << (int)_drv.kernel_map[i].second;
                if (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_stream->close();
            _drv.kernel_id_stream.reset();
        }
    }
}

static ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

static void ggml_backend_et_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;
    if (ctx->data != nullptr) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        if (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte*>(ctx->data));
        }
    }
    delete ctx;
}

static void * ggml_backend_et_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;
    return ctx->data;
}

static enum ggml_status ggml_backend_et_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    if (tensor->view_src != NULL) {
        return GGML_STATUS_SUCCESS;
    }

    const size_t original_size = ggml_nbytes(tensor);
    const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    if (padded_size > original_size) {
        const size_t padding_size = padded_size - original_size;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)buffer->buft->device->context;
        if (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            return GGML_STATUS_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte*>(tensor->data) + original_size;
        if (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_size)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_et_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)buffer->buft->device->context;
    rt::StreamId stream = dev_ctx->default_stream;

    std::byte * dst_ptr = static_cast<std::byte*>(tensor->data) + offset;
    const std::byte * src_ptr = static_cast<const std::byte*>(data);

    rt::EventId event = runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

static void ggml_backend_et_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)buffer->buft->device->context;
    rt::StreamId stream = dev_ctx->default_stream;

    const std::byte * src_ptr = static_cast<const std::byte*>(tensor->data) + offset;
    std::byte * dst_ptr = static_cast<std::byte*>(data);

    rt::EventId event = runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, true /*barrier*/);

    runtime->waitForEvent(event);
}

static bool ggml_backend_et_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_et_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;

    if (ctx->size == 0 || ctx->data == nullptr) {
        return;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)buffer->buft->device->context;
    if (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        return;
    }

    // Use device-side memset kernel for efficient clearing
    if (!ggml_et_memset(dev_ctx, ctx->data, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        return;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

static const struct ggml_backend_buffer_i ggml_backend_et_buffer_i = {
    /* .free_buffer     = */ ggml_backend_et_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_et_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_et_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_et_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_et_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_et_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_et_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_et_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_ET_NAME;
}

static ggml_backend_buffer_t ggml_backend_et_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_et_buffer_type_context * btctx = (ggml_backend_et_buffer_type_context *)buft->context;

    ggml_backend_et_buffer_context * ctx = new ggml_backend_et_buffer_context;
    ctx->devidx = btctx->devidx;
    ctx->size = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        delete ctx;
        return nullptr;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    if (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        delete ctx;
        return nullptr;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->data = runtime->mallocDevice(ctx->rtid, size);
    if (ctx->data == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_et_buffer_i, ctx, size);
}

static size_t ggml_backend_et_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime || !buft->device) {
        return GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)buft->device->context;
    rt::DeviceProperties prop = runtime->getDeviceProperties(dev_ctx->rtid);
    return prop.cacheLineSize_;
}

static size_t ggml_backend_et_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    if (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)buft->device->context;
        return dev_ctx->total_mem;
    }
    return SIZE_MAX;
}

static size_t ggml_backend_et_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes_pad(tensor);
}

static bool ggml_backend_et_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const struct ggml_backend_buffer_type_i ggml_backend_et_buffer_type_i = {
    /* .get_name         = */ ggml_backend_et_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_et_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_et_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_et_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_et_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_et_buffer_type_is_host,
};

static const char * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_ET_NAME;
}

static void ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *)backend->context;

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    if (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;

        if (_drv.profiling_enabled) {
            auto kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);
    }

    delete et_ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_et_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *)backend->context;

    return ggml_backend_et_buffer_type(et_ctx->devidx);
}

static void ggml_backend_et_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)backend->device->context;
    rt::StreamId stream = dev_ctx->default_stream;

    std::byte * dst_ptr = static_cast<std::byte*>(tensor->data) + offset;
    const std::byte * src_ptr = static_cast<const std::byte*>(data);

    runtime->memcpyHostToDevice(stream, src_ptr, dst_ptr, size, true /*barrier*/);
}

static void ggml_backend_et_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)backend->device->context;
    rt::StreamId stream = dev_ctx->default_stream;

    const std::byte * src_ptr = static_cast<const std::byte*>(tensor->data) + offset;
    std::byte * dst_ptr = static_cast<std::byte*>(data);

    runtime->memcpyDeviceToHost(stream, src_ptr, dst_ptr, size, true /*barrier*/);
}

static bool ggml_backend_et_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    if (!runtime) {
        return;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)backend->device->context;
    runtime->waitForStream(dev_ctx->default_stream);

    auto errors = runtime->retrieveStreamErrors(dev_ctx->default_stream);
    if(errors.empty()) {
        return;
    }
    for(const auto& err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (int)err.errorCode_, (int)err.errorContext_.value()[0].type_);
    }
    abort();
}

static bool ggml_et_can_fuse(const struct ggml_cgraph * cgraph, int node_idx,
                             std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 &&
        ops.begin()[0] == GGML_OP_RMS_NORM &&
        ops.begin()[1] == GGML_OP_MUL) {

        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];

        // ET only supports F32
        if (rms_norm->src[0]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        const ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        if (weights->type != GGML_TYPE_F32) {
            return false;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        if (!ggml_is_contiguous(rms_norm->src[0]) ||
            !ggml_is_contiguous_rows(weights)) {
            return false;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        if (rms_norm->src[0]->ne[0] % 16 != 0 ||
            weights->ne[0] % 16 != 0) {
            return false;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        if (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)backend->device->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_NONE) {
            continue;
        }

        // --- Fusion checks (before regular dispatch) ---
        if (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, node, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            continue;
        }

        switch (node->op) {
            case GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, node);
                break;

            case GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, node);
                break;

            case GGML_OP_SUM_ROWS:
                ggml_et_op_sum_rows(dev_ctx, node);
                break;

            case GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, node);
                break;

            case GGML_OP_ADD:
                ggml_et_op_add(dev_ctx, node);
                break;

            case GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, node);
                break;

            case GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, node);

                // if (once < 100){
                //     uint64_t * host_data = (uint64_t *) node->data;

                //     // printf("Tensor error: %lu\n", host_data[0]);

                //     // printf("Tensor error:");
                //     once++;
                // }
                break;

            case GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, node);
                break;

            case GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, node);
                break;

            case GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, node);
                break;

            case GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, node);
                break;

            case GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, node);
                break;

            case GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, node);
                break;

            case GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, node);
                break;

            case GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, node);
                break;

            case GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, node);
                break;

            case GGML_OP_GET_ROWS:
                ggml_et_op_get_rows(dev_ctx, node);
                break;

            case GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, node);
                break;

            case GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, node);
                break;

            case GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, node);
                break;

            case GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, node);
                break;

            case GGML_OP_SET_ROWS:
                ggml_et_op_set_rows(dev_ctx, node);
                break;

            case GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, node);
                break;

            case GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, node);
                break;

            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                // These are metadata-only operations that require no computation
                break;

            default:
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    bool supported = false;
    switch (op->op) {
        case GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_SUM_ROWS:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->src[0]->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_UNARY:
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->ne[0] % 16 == 0 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0])) {
                switch (ggml_get_unary_op(op)) {
                    case GGML_UNARY_OP_ABS:
                    case GGML_UNARY_OP_SGN:
                    case GGML_UNARY_OP_NEG:
                    case GGML_UNARY_OP_STEP:
                    case GGML_UNARY_OP_TANH:
                    case GGML_UNARY_OP_ELU:
                    case GGML_UNARY_OP_RELU:
                    case GGML_UNARY_OP_SIGMOID:
                    case GGML_UNARY_OP_GELU:
                    case GGML_UNARY_OP_GELU_QUICK:
                    case GGML_UNARY_OP_SILU:
                    case GGML_UNARY_OP_HARDSWISH:
                    case GGML_UNARY_OP_HARDSIGMOID:
                    case GGML_UNARY_OP_EXP:
                    case GGML_UNARY_OP_EXPM1:
                    case GGML_UNARY_OP_SOFTPLUS:
                    case GGML_UNARY_OP_GELU_ERF:
                    case GGML_UNARY_OP_FLOOR:
                    case GGML_UNARY_OP_CEIL:
                    case GGML_UNARY_OP_ROUND:
                    case GGML_UNARY_OP_TRUNC:
                        supported = true;
                        break;
                    default:
                        break;
                }
            }
            break;
        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                       op->ne[0] % 16 == 0 && // cache-aligned
                       op->src[0]->ne[0] % 16 == 0 &&
                       (op->src[1]->ne[0] % 16 == 0 || op->src[1]->ne[0] == 1) &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]) &&
                       ggml_is_contiguous(op->src[1]);
            break;
        case GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            if(op->type == GGML_TYPE_F32 && ((op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32)
                    || (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16))
                && op->src[0]->ne[1] % 16 == 0 // m
                && op->src[0]->ne[0] % 16 == 0 &&  // k
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                // Special path for the FP32 TensorFMA kernel
                // Limitation - N is free but the other dims must be multiple of 16
                // The m edge is difficult to do because of the 4 conseqtive load hardware limitation
                // And the k edge is impossible because that is encoded as `stride & 0xFFFFFFFFFFC0ULL` which becomes 0 for stride 16 (4x FP32) :(
                // Can loosen but no real way to make them free. The n edge is the only free one
                // FIXME: Right now this overwrites the mul_mat_f32 kernel - whatever. Fix later. Demo code
                supported = true;
            }
            else if (op->type == GGML_TYPE_F32 &&
                op->src[0] && (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32) {

                // Check first dimension contiguity requirements
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_size(op->src[1]->type));
                bool dst_first_dim_contiguous = (op->nb[0] == sizeof(float));

                // Check destination stride ordering (only for dimensions with ne > 1)
                bool dst_properly_ordered = true;
                for (int d = 0; d < 3; d++) {
                    if (op->ne[d] > 1 && op->ne[d+1] > 1 && op->nb[d] > op->nb[d+1]) {
                        dst_properly_ordered = false;
                    }
                }

                supported = src0_first_dim_contiguous &&
                           src1_first_dim_contiguous &&
                           dst_first_dim_contiguous &&
                           dst_properly_ordered;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2] && op->src[2]->type == GGML_TYPE_I32) {

                // Check first dimension contiguity requirements (matching CPU backend)
                bool src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));
                bool src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_size(op->src[1]->type));
                bool src2_first_dim_contiguous = (op->src[2]->nb[0] == ggml_type_size(op->src[2]->type));
                bool dst_first_dim_contiguous = (op->nb[0] == sizeof(float));

                // Check destination stride ordering (only for dimensions with ne > 1)
                bool dst_properly_ordered = true;
                for (int d = 0; d < 3; d++) {
                    if (op->ne[d] > 1 && op->ne[d+1] > 1 && op->nb[d] > op->nb[d+1]) {
                        dst_properly_ordered = false;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                bool dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                 (op->src[1]->ne[3] == 1) &&  // b is 3d
                                 (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                 (op->src[2]->ne[1] == op->src[1]->ne[2]) &&  // must have expert list per b row
                                 (op->src[0]->ne[0] == op->src[1]->ne[0]) &&  // K dimension must match
                                 (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_first_dim_contiguous &&
                           src1_first_dim_contiguous &&
                           src2_first_dim_contiguous &&
                           dst_first_dim_contiguous &&
                           dst_properly_ordered &&
                           dims_valid;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE (standard and NEOX modes only)
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0])) {
                // Check ROPE mode - only support standard (0x0) and NEOX (0x2)
                const int mode = ((const int32_t *) op->op_params)[2];
                const int ndims = ((const int32_t *) op->op_params)[1];
                supported = ((mode == 0x0) || ((mode & GGML_ROPE_TYPE_NEOX) && ndims % 16 == 0)) && (ndims <= 256);
            } else {
                supported = false;
            }
            break;
        case GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 &&
                       op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(op) &&
                       ggml_is_contiguous(op->src[0]) &&
                       (ggml_nelements(op) % 16 == 0);
            break;
        case GGML_OP_GLU:
            // Support F32 GLU operations (split tensor mode only)
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && // Require split mode
                ggml_nelements(op) % 16 == 0 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                // Check GLU variant - support SWIGLU and GEGLU
                ggml_glu_op glu_type = ggml_get_glu_op(op);
                supported = (glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_GEGLU);
            } else {
                supported = false;
            }
            break;
        case GGML_OP_SOFT_MAX:
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                op->src[0]->ne[0] % 16 == 0) {
                // Check optional mask tensor (F32 only)
                if (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    if (!supported) break;
                }
                // Check optional sinks tensor (F32 only)
                if (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } else {
                    supported = true;
                }
            } else {
                supported = false;
            }
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                op->src[2] && (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) &&
                op->src[4] == nullptr &&
                ggml_is_contiguous_rows(op) &&
                ggml_is_contiguous_rows(op->src[0])) {
                float max_bias = 0.0f;
                float logit_softcap = 0.0f;
                memcpy(&max_bias,      (const float *) op->op_params + 1, sizeof(max_bias));
                memcpy(&logit_softcap, (const float *) op->op_params + 2, sizeof(logit_softcap));

                const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                bool mask_ok = (op->src[3] == nullptr) ||
                               (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                const int64_t nhq = op->src[0]->ne[2];
                const int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                const size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                const size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                supported =
                    mask_ok &&
                    (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                    max_bias == 0.0f &&
                    logit_softcap == 0.0f &&
                    op->src[0]->nb[0] == sizeof(float) &&
                    op->src[1]->nb[0] == k_elem &&
                    op->src[2]->nb[0] == v_elem &&
                    op->nb[0] == sizeof(float) &&
                    op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                    op->src[2]->ne[0] == op->ne[0] &&           // dv matches
                    op->src[2]->ne[0] <= 256 &&                 // dv limit
                    op->src[0]->ne[0] <= 256 &&                 // dk limit
                    nhq % nhk == 0 &&                           // GQA ratio is integer
                    op->src[0]->ne[1] == op->ne[2] &&
                    op->src[0]->ne[2] == op->ne[1] &&
                    op->src[0]->ne[3] == op->ne[3] &&
                    op->src[1]->ne[1] == op->src[2]->ne[1] &&
                    op->src[1]->ne[2] == op->src[2]->ne[2] &&
                    op->src[1]->ne[3] == op->src[2]->ne[3] &&
                    op->src[0]->ne[3] == op->src[1]->ne[3];
            } else {
                supported = false;
            }
            break;
        case GGML_OP_GET_ROWS:
            // Support F32/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_Q4_0 ||
                    op->src[0]->type == GGML_TYPE_Q8_0 ||
                    op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } else {
                supported = false;
            }
            break;
        case GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            if ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                op->src[0] && op->src[0]->type == op->type &&
                ggml_is_contiguous(op) &&
                op->nb[1] % 64 == 0) { // cache alignment
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                if (op->data && op->src[0]->data && op->data == op->src[0]->data) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = false;
                } else {
                    supported = true;
                }
            } else {
                supported = false;
            }
            break;
        case GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            if (op->src[0]) {
                const int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                if (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = true;
                } else if ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                           op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    if (op->data && op->src[0]->data && op->data == op->src[0]->data) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = false;
                    } else {
                        supported = true;
                    }
                } else if (op->type == GGML_TYPE_F16 &&
                           op->src[0]->type == GGML_TYPE_F32 &&
                           ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = true;
                } else {
                    supported = false;
                }
            } else {
                supported = false;
            }
            break;
        case GGML_OP_CONCAT:
            // F32 contiguous, ne[0] cacheline-aligned (16 floats = 64 bytes)
            // For dim==0, both src ne[0] must also be cacheline-aligned
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                op->ne[0] % 16 == 0 &&
                op->src[0]->ne[0] % 16 == 0 &&
                op->src[1]->ne[0] % 16 == 0 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                supported = true;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_REPEAT:
            // F32 contiguous, src0 ne[0] cacheline-aligned (tile unit)
            // dst.ne[i] must be divisible by src0.ne[i] for all dims
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] % 16 == 0 &&
                ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) &&
                op->ne[0] % op->src[0]->ne[0] == 0 &&
                op->ne[1] % op->src[0]->ne[1] == 0 &&
                op->ne[2] % op->src[0]->ne[2] == 0 &&
                op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = true;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) &&
                ggml_is_contiguous(op->src[5])) {
                supported = true;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            if (op->type == GGML_TYPE_F32 &&
                op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&
                op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) &&
                ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = true;
            } else {
                supported = false;
            }
            break;
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = true;
            break;
        case GGML_OP_SET_ROWS:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            if (op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) &&
                ggml_is_contiguous_rows(op) &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&  // same number of columns
                           (op->ne[2] == op->src[0]->ne[2]) &&   // same batch size
                           (op->ne[3] == op->src[0]->ne[3]) &&   // same outer dimension
                           (op->src[0]->ne[1] == op->src[1]->ne[0]) && // src rows = index count
                           (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) && // batch constraint
                           (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) && // outer constraint
                           (op->src[1]->ne[3] == 1);                       // indices tensor constraint
            } else {
                supported = false;
            }
            break;
        case GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = true;
            break;
        default:
            supported = false;
            break;
    }
    if(!supported) {
        ggml_et_dump_operator_metadata(op);
    }
    return supported;
}

static bool ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft->iface.get_name == ggml_backend_et_buffer_type_get_name;
}

static bool ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    if (op->op == GGML_OP_GET_ROWS) {
        return false;
    }
    return true;

    GGML_UNUSED(dev);
}

static const struct ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static const char * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_ET_NAME;
}

static const char * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->desc.c_str();
}

static void ggml_backend_et_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *free = dev_ctx->total_mem;
    *total = dev_ctx->total_mem;
}

static enum ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_et_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    GGML_UNUSED(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id   = NULL;  // No PCI device ID available
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return ggml_backend_et_init(dev_ctx->devidx);
}

static ggml_backend_buffer_type_t ggml_backend_et_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->buftype;
}

static ggml_backend_buffer_type_t ggml_backend_et_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static const struct ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_memory,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ NULL,
    /* .event_free        = */ NULL,
    /* .event_synchronize = */ NULL,
};


/*
  Backend Registry.
*/

static const char * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_ET_NAME;
}

static size_t ggml_backend_et_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *)reg->context;
    if (devidx >= ctx->devices.size()) {
        return nullptr;
    }
    return ctx->devices[devidx];
}

static void * ggml_backend_et_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_count,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_address,
};

ggml_backend_reg_t ggml_backend_et_reg(void) {
    static ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
	ggml_backend_et_reg_ctx * ctx = new ggml_backend_et_reg_ctx;

	if (!ggml_et_driver_init())
	    return nullptr;

	ggml_backend_reg_t r = new ggml_backend_reg {
	    /* .api_version = */ GGML_BACKEND_API_VERSION,
	    /* .iface       = */ ggml_backend_et_reg_i,
	    /* .context     = */ nullptr, // Set later
	};

	std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        for (int i = 0; i < ggml_et_devicelayer()->getDevicesCount(); i++) {
	    ggml_backend_dev_t dev = new ggml_backend_device {
		/* .iface   = */ ggml_backend_et_device_i,
		/* .reg     = */ r,
		/* .context = */ nullptr // Set later
	    };

	    rt::DeviceId rtid = rtids[i];
	    rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

	    // Create device context.
	    ggml_backend_et_device_context * dev_ctx = new ggml_backend_et_device_context;
	    dev_ctx->devidx = i;
	    dev_ctx->rtid = rtid;
	    dev_ctx->name = GGML_ET_NAME + std::to_string(i);
	    dev_ctx->desc = "ET device " + std::to_string(i);
	    dev_ctx->total_mem = static_cast<size_t>(prop.memorySize_);
	    // Add buffer type for device to device context.
	    ggml_backend_et_buffer_type_context * bufty_ctx = new ggml_backend_et_buffer_type_context;
	    bufty_ctx->devidx = i;
	    bufty_ctx->name = GGML_ET_NAME + std::to_string(i);
	    dev_ctx->buftype = new ggml_backend_buffer_type {
		/* .iface   = */ ggml_backend_et_buffer_type_i,
		/* .device  = */ dev,
		/* .context = */ bufty_ctx
	    };

	    // Create default stream for ordered execution on this device
	    dev_ctx->default_stream = ggml_et_runtime()->createStream(rtid);

		dev_ctx->trace_buffer = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_BUFFER_SIZE);

	    dev->context = dev_ctx;

	    ctx->devices.push_back(dev);
	}

	r->context = ctx;
	return r;
    }();

    return _reg;
}

ggml_guid_t ggml_backend_et_guid(void) {
    static ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4, 0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    return &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    if (!ggml_et_driver_init()) {
        return nullptr;
    }

    if (devidx >= (size_t)ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_et_context * ctx = new ggml_backend_et_context;
    ctx->devidx = (int)devidx;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_et(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

int ggml_backend_et_get_device_count(void) {
    return ggml_backend_et_reg_get_device_count(ggml_backend_et_reg());
}

void ggml_backend_et_get_device_description(int devidx, char * description, size_t description_size) {
    if (devidx < 0 || devidx >= ggml_backend_et_get_device_count()) {
        snprintf(description, description_size, "ET Device %d (invalid)", devidx);
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    snprintf(description, description_size, "%s", dev_ctx->desc.c_str());
}

void ggml_backend_et_get_device_memory(int devidx, size_t * free, size_t * total) {
    if (devidx < 0 || devidx >= ggml_backend_et_get_device_count()) {
        *free = 0;
        *total = 0;
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_memory(dev, free, total);
}

ggml_backend_buffer_type_t ggml_backend_et_buffer_type(size_t dev_num) {
    if (dev_num >= (size_t)ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_num);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->buftype;
}

ggml_backend_buffer_type_t ggml_backend_et_host_buffer_type(void) {
    static ggml_backend_buffer_type host_buffer_type = {
        /* .iface   = */ ggml_backend_et_buffer_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &host_buffer_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)
