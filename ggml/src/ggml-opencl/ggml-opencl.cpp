#include "cl-common.h"
#include "ops.h"

static ggml_backend_opencl_context * ggml_cl_init(ggml_backend_dev_t dev);
static bool ggml_opencl_is_device_supported(ggml_backend_dev_t dev);

static std::vector<ggml_backend_device> ggml_opencl_probe_devices(ggml_backend_reg * reg) {
    std::vector<ggml_backend_device> found_devices;

    struct cl_device;
    struct cl_platform {
        cl_platform_id id;
        unsigned number;
        char name[128];
        char vendor[128];
        struct cl_device * devices;
        unsigned n_devices;
        struct cl_device * default_device;
    };

    struct cl_device {
        struct cl_platform * platform;
        cl_device_id id;
        unsigned number;
        cl_device_type type;
        char name[128];
        char version[128];
    };

    enum { NPLAT = 16, NDEV = 16 };

    struct cl_platform platforms[NPLAT];
    unsigned n_platforms = 0;
    struct cl_device devices[NDEV];
    unsigned n_devices = 0;
    struct cl_device * default_device = NULL;
    unsigned           default_platform_number = 0;

    cl_platform_id platform_ids[NPLAT];
    if (clGetPlatformIDs(NPLAT, platform_ids, &n_platforms) != CL_SUCCESS) {
        GGML_LOG_ERROR("ggml_opencl: platform IDs not available.\n");
        return found_devices;
    }

    for (unsigned i = 0; i < n_platforms; i++) {
        struct cl_platform * p = &platforms[i];
        p->number = i;
        p->id = platform_ids[i];
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_NAME, sizeof(p->name), &p->name, NULL));
        CL_CHECK(clGetPlatformInfo(p->id, CL_PLATFORM_VENDOR, sizeof(p->vendor), &p->vendor, NULL));

        cl_device_id device_ids[NDEV];
        cl_int clGetDeviceIDsError = clGetDeviceIDs(p->id, CL_DEVICE_TYPE_ALL, NDEV, device_ids, &p->n_devices);
        if (clGetDeviceIDsError == CL_DEVICE_NOT_FOUND) {
            p->n_devices = 0;
        } else {
            CL_CHECK(clGetDeviceIDsError);
        }
        p->devices = p->n_devices > 0 ? &devices[n_devices] : NULL;
        p->default_device = NULL;

        for (unsigned j = 0; j < p->n_devices; j++) {
            struct cl_device * d = &devices[n_devices];
            d->number = n_devices++;
            d->id = device_ids[j];
            d->platform = p;
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_NAME, sizeof(d->name), &d->name, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_TYPE, sizeof(d->type), &d->type, NULL));
            CL_CHECK(clGetDeviceInfo(d->id, CL_DEVICE_VERSION, sizeof(d->version), &d->version, NULL));

            if (p->default_device == NULL && d->type == CL_DEVICE_TYPE_GPU) {
                p->default_device = d;
            }
        }

        if (default_device == NULL && p->default_device != NULL) {
            default_device          = p->default_device;
            default_platform_number = i;
        }
    }

    if (n_devices == 0) {
        GGML_LOG_ERROR("ggml_opencl: could find any OpenCL devices.\n");
        return found_devices;
    }

    char *      user_platform_string = getenv("GGML_OPENCL_PLATFORM");
    char *      user_device_string   = getenv("GGML_OPENCL_DEVICE");
    int         user_platform_number = -1;
    int         user_device_number   = -1;
    cl_device * candidate_devices    = nullptr;
    unsigned    n_candidate_devices  = 0;

    unsigned n;
    if (user_platform_string != NULL && sscanf(user_platform_string, " %u", &n) == 1 && n < n_platforms) {
        user_platform_number = (int)n;
    }
    if (user_device_string != NULL && sscanf(user_device_string, " %u", &n) == 1 && n < n_devices) {
        user_device_number = (int)n;
    }
    if (user_platform_number != -1 && user_device_number != -1) {
        cl_platform* platform = &platforms[user_platform_number];
        if ((unsigned)user_device_number >= platform->n_devices) {
            GGML_LOG_ERROR("ggml_opencl: invalid device number %d\n", user_device_number);
            exit(1);
        }
        default_device      = &platform->devices[user_device_number];
        candidate_devices   = platform->devices;
        n_candidate_devices = platform->n_devices;
    } else {
        // Choose a platform by matching a substring.
        if (user_platform_number == -1 && user_platform_string != NULL && user_platform_string[0] != 0) {
            for (unsigned i = 0; i < n_platforms; i++) {
                struct cl_platform * p = &platforms[i];
                if (strstr(p->name, user_platform_string) != NULL ||
                    strstr(p->vendor, user_platform_string) != NULL) {
                    user_platform_number = (int)i;
                    break;
                }
            }
            if (user_platform_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no platform matching '%s' was found.\n", user_platform_string);
                exit(1);
            }
        }

        int                  platform_idx = user_platform_number != -1 ? user_platform_number : default_platform_number;
        struct cl_platform * p            = &platforms[platform_idx];
        candidate_devices                 = p->devices;
        n_candidate_devices               = p->n_devices;
        default_device                    = p->default_device;
        if (n_candidate_devices == 0) {
            GGML_LOG_ERROR("ggml_opencl: selected platform '%s' does not have any devices.\n", p->name);
            exit(1);
        }

        if (user_device_number == -1 && user_device_string != NULL && user_device_string[0] != 0) {
            for (unsigned i = 0; i < n_candidate_devices; i++) {
                struct cl_device * d = &candidate_devices[i];
                if (strstr(d->name, user_device_string) != NULL) {
                    user_device_number = d->number;
                    break;
                }
            }
            if (user_device_number == -1) {
                GGML_LOG_ERROR("ggml_opencl: no device matching '%s' was found.\n", user_device_string);
                exit(1);
            }
        }
        if (user_device_number != -1) {
            candidate_devices   = &devices[user_device_number];
            n_candidate_devices = 1;
            default_device      = &candidate_devices[0];
        }

        GGML_ASSERT(n_candidate_devices > 0);

        if (default_device == NULL) {
            default_device = &candidate_devices[0];
        }
    }

    GGML_ASSERT(n_candidate_devices != 0 && candidate_devices);

    // Put the default device in front.
    for (unsigned i = 1; i < n_candidate_devices; i++) {
        if (&candidate_devices[i] == default_device) {
            std::swap(candidate_devices[0], candidate_devices[i]);
            default_device = &candidate_devices[0];
            break;
        }
    }

    GGML_LOG_INFO("ggml_opencl: selected platform: '%s'\n", default_device->platform->name);

    std::vector<cl_device_id> device_ids;
    for (auto dev = candidate_devices, dev_end = candidate_devices + n_candidate_devices; dev != dev_end; dev++) {
        device_ids.push_back(dev->id);
    }

    cl_int                err;
    cl_context            shared_context;
    cl_context_properties properties[] = { (intptr_t) CL_CONTEXT_PLATFORM, (intptr_t) default_device->platform->id, 0 };

    CL_CHECK(
        (shared_context = clCreateContext(properties, device_ids.size(), device_ids.data(), NULL, NULL, &err), err));

    for (auto dev = candidate_devices, dev_end = candidate_devices + n_candidate_devices; dev != dev_end; dev++) {
        GGML_LOG_INFO("\nggml_opencl: device: '%s (%s)'\n", dev->name, dev->version);

        auto dev_ctx = std::unique_ptr<ggml_backend_opencl_device_context>(new ggml_backend_opencl_device_context{
            /*.platform         =*/dev->platform->id,
            /*.platform_nane    =*/dev->platform->name,
            /*.device           =*/dev->id,
            /*.device_name      =*/dev->name,
            /*.device_type      =*/dev->type,
            /*.device_version   =*/dev->version,
            /*.backend_ctx      =*/nullptr,
            /*.buffer_type      =*/{},
            /*.context          =*/shared_context,
        });

        found_devices.push_back(ggml_backend_device{
            /* .iface   = */ ggml_backend_opencl_device_i,
            /* .reg     = */ reg,
            /* .context = */ dev_ctx.get(),
        });

        if (!ggml_opencl_is_device_supported(&found_devices.back())) {
            found_devices.pop_back();
            GGML_LOG_WARN("ggml_opencl: drop unsupported device '%s'.\n", dev->name);
            continue;
        }

        g_ggml_backend_opencl_dev_ctxs.push_back(std::move(dev_ctx));
    }

    if (found_devices.size()) {
        auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(found_devices.front().context);
        GGML_LOG_INFO("ggml_opencl: default device: '%s (%s)'\n", dev_ctx->device_name.c_str(),
                      dev_ctx->device_version.c_str());

        if (dev_ctx->device_type != CL_DEVICE_TYPE_GPU) {
            GGML_LOG_WARN("ggml_opencl: warning, the default device is not a GPU: '%s'.\n",
                          dev_ctx->device_name.c_str());
        }
    }

    return found_devices;
}

static void ggml_opencl_print_backend_info(ggml_backend_opencl_device_context * dev_ctx) {
    GGML_ASSERT(dev_ctx);
    GGML_ASSERT(dev_ctx->backend_ctx);

    auto * backend_ctx = dev_ctx->backend_ctx;

    GGML_LOG_INFO("ggml_opencl: OpenCL device: %s\n",
        backend_ctx->device_name.c_str());
    GGML_LOG_INFO("ggml_opencl: OpenCL driver: %s\n",
        backend_ctx->driver_version.c_str());
    GGML_LOG_INFO("ggml_opencl: vector subgroup broadcast support: %s\n",
        backend_ctx->has_vector_subgroup_broadcast ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: subgroup shuffle support: %s\n",
        backend_ctx->has_subgroup_shuffle ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: device FP16 support: %s\n",
        backend_ctx->fp16_support ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: khr dot product support: %s\n",
        backend_ctx->has_integer_dot ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: mem base addr align: %u\n",
        backend_ctx->alignment);
    GGML_LOG_INFO("ggml_opencl: global mem size: %zu MB\n",
        backend_ctx->global_mem_size/1024/1024);
    GGML_LOG_INFO("ggml_opencl: max mem alloc size: %zu MB\n",
        backend_ctx->max_alloc_size/1024/1024);
    GGML_LOG_INFO("ggml_opencl: device max image buffer size (pixels): %zu\n",
        backend_ctx->image_max_buffer_size);
    GGML_LOG_INFO("ggml_opencl: device max image2d size: %zu x %zu\n",
        backend_ctx->image2d_max_width, backend_ctx->image2d_max_height);
    GGML_LOG_INFO("ggml_opencl: device max workgroup size: %zu\n",
        backend_ctx->max_workgroup_size);
    GGML_LOG_INFO("ggml_opencl: SVM coarse grain buffer support: %s\n",
        backend_ctx->svm_caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain buffer support: %s\n",
        backend_ctx->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM fine grain system support: %s\n",
        backend_ctx->svm_caps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: SVM atomics support: %s\n",
        backend_ctx->svm_caps & CL_DEVICE_SVM_ATOMICS ? "true" : "false");
    GGML_LOG_INFO("ggml_opencl: cl_qcom_subgroup_shuffle support: %s\n",
        backend_ctx->has_qcom_subgroup_shuffle ? "true" : "false");

    // Print out configurations
#ifdef GGML_OPENCL_SOA_Q
    GGML_LOG_INFO("ggml_opencl: flattening quantized weights representation as struct of arrays (GGML_OPENCL_SOA_Q)\n");
#endif // GGML_OPENCL_SOA_Q

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    GGML_LOG_INFO("ggml_opencl: using kernels optimized for Adreno (GGML_OPENCL_USE_ADRENO_KERNELS)\n");
    if (backend_ctx->adreno_xmem_gemm_enabled) {
        GGML_LOG_INFO("ggml_opencl: Adreno xmem F16xF32 GEMM enabled (temporary weight prepack)\n");
    }
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    if (backend_ctx->adreno_use_large_buffer) {
        if (!backend_ctx->adreno_has_large_buffer) {
            GGML_LOG_INFO("ggml_opencl: Adreno large buffer requested but not supported by driver, will use regular buffer\n");
            backend_ctx->adreno_use_large_buffer = false;
        } else {
            GGML_LOG_INFO("ggml_opencl: Adreno large buffer enabled\n");
        }
    }

    if (dev_ctx->opfilter) {
        // for information only, the actual regex object is created in ggml_opencl_is_device_supported
        GGML_LOG_INFO("ggml_opencl: opfilter regex = \"%s\"\n", dev_ctx->opfilter_str.c_str());
    }

#ifdef GGML_OPENCL_PROFILING
    GGML_LOG_INFO("ggml_opencl: OpenCL profiling enabled\n");
#endif
}

// check if device should be accepted
static bool ggml_opencl_is_device_supported(ggml_backend_dev_t dev) {
    GGML_ASSERT(dev);
    GGML_ASSERT(dev->context);

    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    GGML_ASSERT(dev_ctx->platform);
    GGML_ASSERT(dev_ctx->device);

    if (strstr(dev_ctx->device_name.c_str(), "Adreno") ||
        strstr(dev_ctx->device_name.c_str(), "Qualcomm") ||
        strstr(dev_ctx->device_version.c_str(), "Adreno")) {
        dev_ctx->gpu_family = GPU_FAMILY::ADRENO;

        // Usually device version contains the detailed device name
        dev_ctx->adreno_gen = get_adreno_gpu_gen(dev_ctx->device_version.c_str());
        if (dev_ctx->adreno_gen == ADRENO_GPU_GEN::ADRENO_UNKNOWN) {
            dev_ctx->adreno_gen = get_adreno_gpu_gen(dev_ctx->device_name.c_str());
        }
    } else if (strstr(dev_ctx->device_name.c_str(), "Intel")) {
        dev_ctx->gpu_family = GPU_FAMILY::INTEL;
    } else {
        GGML_LOG_WARN("ggml_opencl: unsupported GPU '%s'.\n", dev_ctx->device_name.c_str());
        dev_ctx->gpu_family = GPU_FAMILY::UNKNOWN;
        return false;
    }

    ggml_cl_version platform_version = get_opencl_platform_version(dev_ctx->platform);

    // Check device OpenCL version, OpenCL 2.0 or above is required
    ggml_cl_version opencl_c_version = get_opencl_c_version(platform_version, dev_ctx->device);
    if (opencl_c_version.major < 2) {
        GGML_LOG_WARN("ggml_opencl: OpenCL 2.0 or above is required\n");
        return false;
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    if (dev_ctx->gpu_family != GPU_FAMILY::ADRENO) {
        GGML_LOG_WARN("ggml_opencl: Adreno-specific kernels should not be enabled for non-Adreno GPUs; "
            "run on an Adreno GPU or recompile with CMake option `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`\n");
        return false;
    }
#endif

    size_t ext_str_size;
    clGetDeviceInfo(dev_ctx->device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);

    char *ext_buffer = (char *)alloca(ext_str_size + 1);
    clGetDeviceInfo(dev_ctx->device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
    ext_buffer[ext_str_size] = '\0';

    // Check if ext_buffer contains cl_khr_fp16
    bool fp16_support = strstr(ext_buffer, "cl_khr_fp16") != NULL;
    if (!fp16_support) {
        GGML_LOG_WARN("ggml_opencl: device does not support FP16\n");
        return false;
    }

    // If OpenCL 3.0 is supported, then check for cl_khr_subgroups, which becomes
    // optional in OpenCL 3.0 (cl_khr_subgroup is mandatory in OpenCL 2.x)
    if (opencl_c_version.major == 3 && strstr(ext_buffer, "cl_khr_subgroups") == NULL &&
        strstr(ext_buffer, "cl_intel_subgroups") == NULL) {
        GGML_LOG_WARN("ggml_opencl: device does not support subgroups (cl_khr_subgroups or cl_intel_subgroups) "
            "(note that subgroups is an optional feature in OpenCL 3.0)\n");
        return false;
    }

    clGetDeviceInfo(dev_ctx->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &dev_ctx->global_mem_size, NULL);

    const char * str_opfilter = getenv("GGML_OPENCL_OPFILTER");
    if (str_opfilter) {
        dev_ctx->opfilter_str = str_opfilter;
        dev_ctx->opfilter = new std::regex(str_opfilter, std::regex_constants::icase);
    }

    return true;
}

// Initialize device if it is supported (returns nullptr if it is not).
static ggml_backend_opencl_context * ggml_cl_init(ggml_backend_dev_t dev) {
    GGML_ASSERT(dev);
    GGML_ASSERT(dev->context);

    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    GGML_ASSERT(dev_ctx->platform);
    GGML_ASSERT(dev_ctx->device);

    if (dev_ctx->backend_ctx) {
        return dev_ctx->backend_ctx;
    }

    auto backend_ctx        = std::make_unique<ggml_backend_opencl_context>();
    backend_ctx->device     = dev_ctx->device;
    backend_ctx->gpu_family = GPU_FAMILY::UNKNOWN;

    // ref_count get increased in ggml_backend_opencl_device_init
    // This function is also used to retrieve backend context, so we don't want
    // to increase ref_count for each call. We only want to increase ref_count
    // when the associated device is initialized
    backend_ctx->ref_count  = 0;

    backend_ctx->gpu_family = dev_ctx->gpu_family;
    backend_ctx->adreno_gen = dev_ctx->adreno_gen;
    if (backend_ctx->gpu_family == GPU_FAMILY::ADRENO) {
        ggml_cl_init_fa_dims_table();

        // Use wave size of 64 for all Adreno GPUs.
        backend_ctx->adreno_wave_size = 64;
    }

    // Populate backend device name
    backend_ctx->device_name = dev_ctx->device_name;

    // A local ref of cl_device_id for convenience
    cl_device_id device = backend_ctx->device;

    ggml_cl_version platform_version = get_opencl_platform_version(dev_ctx->platform);
    ggml_cl_version opencl_c_version = get_opencl_c_version(platform_version, device);

    backend_ctx->platform_version = platform_version;
    backend_ctx->opencl_c_version = opencl_c_version;

    // Check driver version
    size_t driver_version_str_size;
    clGetDeviceInfo(device, CL_DRIVER_VERSION, 0, NULL, &driver_version_str_size);
    char *driver_version = (char *)alloca(driver_version_str_size + 1);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, driver_version_str_size, driver_version, NULL);
    driver_version[driver_version_str_size] = '\0';
    backend_ctx->driver_version = driver_version;

    backend_ctx->adreno_cl_compiler_version = get_adreno_cl_compiler_version(driver_version);
    backend_ctx->has_vector_subgroup_broadcast =
        (backend_ctx->adreno_cl_compiler_version.type == E031 && backend_ctx->adreno_cl_compiler_version.major >= 47) ||
        (backend_ctx->adreno_cl_compiler_version.type == DX   && backend_ctx->adreno_cl_compiler_version.major >= 17);

    // The q6_K flat mul_mat miscompile is a defect of the older E031 compilers, not a
    // property of any GPU generation: it reproduces on E031.38 (Adreno 642L) and E031.41
    // (Adreno 740) and is fixed by E031.45 (Adreno 619). Gate on the compiler so parts
    // that do not need the workarounds do not pay for them. The explicit type check is
    // required: newer_than_or_same() is false for every non-E031 compiler, so negating it
    // alone would enable the workarounds on E17/DX.
    backend_ctx->q6_k_flat_old_compiler =
        backend_ctx->adreno_cl_compiler_version.type == E031 &&
        !backend_ctx->adreno_cl_compiler_version.newer_than_or_same(E031, 45, 0, 0);

    size_t ext_str_size;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_str_size);
    char *ext_buffer = (char *)alloca(ext_str_size + 1);
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_str_size, ext_buffer, NULL);
    ext_buffer[ext_str_size] = '\0'; // ensure it is null terminated

    // check support for qcom_subgroup_shuffle
    if (strstr(ext_buffer, "cl_qcom_subgroup_shuffle") != NULL) {
        backend_ctx->has_qcom_subgroup_shuffle = true;
    }

    // Check if ext_buffer contains cl_khr_fp16
    backend_ctx->fp16_support = strstr(ext_buffer, "cl_khr_fp16") != NULL;

    // check Adreno large buffer support
    backend_ctx->adreno_has_large_buffer = strstr(ext_buffer, "cl_qcom_large_buffer") != NULL;

    // subgroup shuffle support (N_SPLIT>1 FA kernel)
    backend_ctx->has_qcom_subgroup_shuffle = strstr(ext_buffer, "cl_qcom_subgroup_shuffle") != NULL;
    backend_ctx->has_subgroup_shuffle =
        strstr(ext_buffer, "cl_khr_subgroup_shuffle") != NULL ||
        backend_ctx->has_qcom_subgroup_shuffle;

    // check for cl_khr_integer_dot_product
    // cl_qcom_dot_product8 uses signed * unsigned
    // while cl_khr_integer_dot_product uses signed * signed -- we stick with khr for now
    backend_ctx->has_integer_dot =
        strstr(ext_buffer, "cl_khr_integer_dot_product") != NULL;

    cl_uint base_align_in_bits;
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &base_align_in_bits, NULL));
    GGML_ASSERT(base_align_in_bits % 8u == 0);
    backend_ctx->alignment = base_align_in_bits / 8u;

    backend_ctx->global_mem_size = dev_ctx->global_mem_size;

    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &backend_ctx->max_alloc_size, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, sizeof(size_t), &backend_ctx->image_max_buffer_size, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &backend_ctx->image2d_max_width, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &backend_ctx->image2d_max_height, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &backend_ctx->max_workgroup_size, NULL));
    CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities), &backend_ctx->svm_caps, 0));

    if (opencl_c_version.major >= 3) {
        // Assume it is not available for 3.0, since it is optional in 3.0.
        // If compiling against 3.0, then we can query.
        backend_ctx->non_uniform_workgroups = false;
#if CL_TARGET_OPENCL_VERSION >= 300
        CL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, sizeof(cl_bool),
                                 &backend_ctx->non_uniform_workgroups, 0));
#endif
    } else {
        GGML_ASSERT(opencl_c_version.major == 2);
        // Non-uniform workgroup sizes is mandatory feature in v2.x.
        backend_ctx->non_uniform_workgroups = true;
    }

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Adreno xmem F16xF32 GEMM, default on adreno, opt out with GGML_OPENCL_ADRENO_XMEM_GEMM=0.
    // This helps models with f16 attention weights, e.g., gpt-oss-20b-f16
    {
        const char * xmem_env = getenv("GGML_OPENCL_ADRENO_XMEM_GEMM");
        backend_ctx->adreno_xmem_gemm_enabled = backend_ctx->gpu_family == GPU_FAMILY::ADRENO &&
                                                (xmem_env ? atoi(xmem_env) != 0 : true);
    }
#endif

    // determine whether to use large buffer for Adreno
    backend_ctx->adreno_use_large_buffer = getenv("GGML_OPENCL_ADRENO_USE_LARGE_BUFFER") != nullptr &&
                                           backend_ctx->gpu_family == GPU_FAMILY::ADRENO;

    // ragged moe, unspecified or non-zero means enabled, set to 0 to disable
    static const char * ragged_fp16_env = getenv("GGML_OPENCL_MOE_RAGGED_FP16");
    backend_ctx->adreno_use_moe_ragged = (ragged_fp16_env == NULL) ? 1 : (atoi(ragged_fp16_env) != 0);

    // ragged moe, tile-skip granularity (columns per skip-group): 8 = quarter (default),
    // 16 = half (legacy), 32 = disabled. Override with GGML_OPENCL_MOE_RAGGED_GRAN={8,16,32}
    static const char * ragged_gran_env = getenv("GGML_OPENCL_MOE_RAGGED_GRAN");
    backend_ctx->adreno_moe_ragged_skip_gran = (ragged_gran_env != NULL) ? atoi(ragged_gran_env) : 8;

    // whether fuse moe combine
    static const char * fuse_moe_bias_glu_env = getenv("GGML_OPENCL_FUSE_MOE_BIAS_GLU");
    backend_ctx->fuse_moe_bias_glu = fuse_moe_bias_glu_env == NULL ? 1 : (atoi(fuse_moe_bias_glu_env) != 0);

    static const char * fuse_moe_bias_combine_env = getenv("GGML_OPENCL_FUSE_MOE_BIAS_COMBINE");
    backend_ctx->fuse_moe_bias_combine = fuse_moe_bias_combine_env == NULL ? 1 : (atoi(fuse_moe_bias_combine_env) != 0);

    static const char * fuse_moe_combine_env = getenv("GGML_OPENCL_FUSE_MOE_COMBINE");
    backend_ctx->fuse_moe_combine = fuse_moe_combine_env == NULL ? 1 : (atoi(fuse_moe_combine_env) != 0);

    // ragged moe dp4 variant
    static const char * ragged_dp4_env = getenv("GGML_OPENCL_MOE_RAGGED");
    backend_ctx->adreno_use_moe_ragged_dp4 = ragged_dp4_env == NULL ? 1 : (atoi(ragged_dp4_env) != 0);

#ifdef GGML_OPENCL_USE_ADRENO_BIN_KERNELS
    // try loading adreno binary kernels if enabled
    // if fails to load, builtin kernels will be used
    {
        dl_handle * kernel_lib_handle = dl_load_library(KERNEL_LIB_NAME);
        backend_ctx->adreno_use_bin_kernels = false;

        if (kernel_lib_handle) {
            backend_ctx->get_adreno_bin_kernel_func = (get_adreno_bin_kernel_func_t)dl_get_sym(kernel_lib_handle, "get_adreno_kernels");
            if (backend_ctx->get_adreno_bin_kernel_func) {
                GGML_LOG_INFO("ggml_opencl: loaded bin kernel library %s\n", KERNEL_LIB_NAME);
                backend_ctx->adreno_use_bin_kernels = true;
            } else {
                GGML_LOG_INFO("ggml_opencl: bin kernel library %s is invalid, will use builtin kernels\n", KERNEL_LIB_NAME);
            }
        } else {
            GGML_LOG_INFO("ggml_opencl: failed to load %s, will use builtin kernels\n", KERNEL_LIB_NAME);
        }
    }
#endif // GGML_OPENCL_USE_ADRENO_BIN_KERNELS

    cl_int err;

    // A local ref of cl_context for convenience
    cl_context context = backend_ctx->context = dev_ctx->context;

    //CL_CHECK((queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err),
    //    (err != CL_INVALID_QUEUE_PROPERTIES && err != CL_INVALID_VALUE ? err :
    //    (queue = clCreateCommandQueue(context, device, 0, &err), err)
    //)));
    cl_command_queue_properties command_queue_props = 0;
#ifdef GGML_OPENCL_PROFILING
    command_queue_props |= CL_QUEUE_PROFILING_ENABLE;
#endif
    CL_CHECK((backend_ctx->queue = clCreateCommandQueue(context, device, command_queue_props, &err), err));

    // delay kernel loading until the first buffer is created
    // load_cl_kernels(backend_ctx.get());

#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
    // Allocate intermediate buffers and images
    size_t required_A_q_d_bytes = 311164928;
    size_t required_A_s_d_bytes = 38895616;
    size_t required_B_d_bytes = 45088768;

    // Ensure buffer sizes do not exceed the maximum allocation size
    size_t max_A_q_d_bytes = MIN(required_A_q_d_bytes, backend_ctx->max_alloc_size);
    size_t max_A_s_d_bytes = MIN(required_A_s_d_bytes, backend_ctx->max_alloc_size);
    size_t max_B_d_bytes   = MIN(required_B_d_bytes, backend_ctx->max_alloc_size);
    if (required_A_q_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: A_q_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_A_q_d_bytes, max_A_q_d_bytes);
    }
    if (required_A_s_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: A_s_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_A_s_d_bytes, max_A_s_d_bytes);
    }
    if (required_B_d_bytes > backend_ctx->max_alloc_size) {
        GGML_LOG_WARN("ggml_opencl: B_d buffer size reduced from %zu to %zu due to device limitations.\n",
                      required_B_d_bytes, max_B_d_bytes);
    }

    backend_ctx->prealloc_quant_trans.allocate(context, max_A_q_d_bytes);
    backend_ctx->prealloc_scales_trans.allocate(context, max_A_s_d_bytes);
    backend_ctx->prealloc_act_trans.allocate(context, max_B_d_bytes);
#endif // GGML_OPENCL_USE_ADRENO_KERNELS

    backend_ctx->disable_fusion = getenv("GGML_OPENCL_DISABLE_FUSION") != nullptr;
    if (const char * env = getenv("GGML_OPENCL_FUSE_MM_GLU")) {
        backend_ctx->fuse_mm_glu = atoi(env) != 0;
    }
    if (const char * env = getenv("GGML_OPENCL_FUSE_RMS_ADD")) {
        backend_ctx->fuse_rms_add = atoi(env) != 0;
    }
    if (const char * env = getenv("GGML_OPENCL_F16_MROW")) {
        backend_ctx->f16_mrow = atoi(env) != 0;
    }
    if (const char * env = getenv("GGML_OPENCL_F16_MROW_RPT")) {
        const int v = atoi(env);
        backend_ctx->f16_mrow_rpt = (v == 2 || v == 4 || v == 8 || v == 16) ? v : 1;
    }

    dev_ctx->backend_ctx = backend_ctx.release();
    return dev_ctx->backend_ctx;
}

static void ggml_cl_free(ggml_backend_t backend) {
    ggml_backend_opencl_context * ctx = (ggml_backend_opencl_context *) backend->context;
    ctx->free();
}

//------------------------------------------------------------------------------
// Backend API
//------------------------------------------------------------------------------

//
// backend
//
static const char * ggml_backend_opencl_name(ggml_backend_t backend) {
    return "OpenCL";

    UNUSED(backend);
}

static void ggml_backend_opencl_free(ggml_backend_t backend) {
    ggml_cl_free(backend);
}

static void ggml_backend_opencl_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_opencl_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_opencl_cpy_tensor_async(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_opencl_synchronize(ggml_backend_t backend) {
    auto * backend_ctx = static_cast<ggml_backend_opencl_context *>(backend->context);

    cl_event evt;
    CL_CHECK(clEnqueueBarrierWithWaitList(backend_ctx->queue, 0, nullptr, &evt));
    CL_CHECK(clWaitForEvents(1, &evt));
    CL_CHECK(clReleaseEvent(evt));
}

// Synchronizes the 'backend_ctx's device with others so that commands
// enqueued to it won't start until commands in the other devices have
// completed.
static bool ggml_cl_tensors_overlap(const ggml_tensor * x, const ggml_tensor * y) {
    ggml_tensor_extra_cl * ex = (ggml_tensor_extra_cl *)x->extra;
    ggml_tensor_extra_cl * ey = (ggml_tensor_extra_cl *)y->extra;
    if (!ex || !ey || ex->data_device != ey->data_device) { return false; }
    const cl_ulong xo = ex->offset + x->view_offs, xe = xo + ggml_nbytes(x);
    const cl_ulong yo = ey->offset + y->view_offs, ye = yo + ggml_nbytes(y);
    return xo < ye && yo < xe;
}

// Detect the MoE combine epilogue: router-weight MUL ([n_embd,k,nt] * [1,k,nt]) followed
// by k VIEWs of it and a (k-1)-long ADD reduction chain producing [n_embd, nt]. When it
// matches (and the output does not alias the inputs), the whole subgraph collapses to one
// weighted-sum-across-experts kernel.
static bool ggml_opencl_can_fuse_moe_combine(const struct ggml_cgraph * cgraph, int node_idx,
                                             const ggml_tensor ** out_final_add) {
    const ggml_tensor * mul = cgraph->nodes[node_idx];
    if (mul->op != GGML_OP_MUL) { return false; }
    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    if (!experts || !weights) { return false; }
    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) { return false; }

    const int64_t n_embd = experts->ne[0];
    const int64_t k      = experts->ne[1];
    const int64_t nt     = experts->ne[2];
    if (k < 2 || k > 64 || experts->ne[3] != 1 || n_embd % 4 != 0) { return false; }
    if (weights->ne[0] != 1 || weights->ne[1] != k || weights->ne[2] != nt || weights->ne[3] != 1) { return false; }
    if (mul->ne[0] != n_embd || mul->ne[1] != k || mul->ne[2] != nt) { return false; }
    // the fused kernel needs contiguous experts/weights and a contiguous 2D dst
    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights)) { return false; }

    const int n_nodes = 1 + (int)k + (int)(k - 1);  // MUL + k*VIEW + (k-1)*ADD
    if (n_nodes >= 32) { return false; }
    if (node_idx + n_nodes > cgraph->n_nodes) { return false; }

    enum ggml_op ops[1 + 64 + 63];
    int n = 0;
    ops[n++] = GGML_OP_MUL;
    for (int j = 0; j < (int)k;     ++j) { ops[n++] = GGML_OP_VIEW; }
    for (int j = 0; j < (int)k - 1; ++j) { ops[n++] = GGML_OP_ADD;  }
    const int outs[] = { node_idx + n_nodes - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, n_nodes, ops, outs, 1)) { return false; }

    for (int j = 0; j < (int)k; ++j) {
        const ggml_tensor * vw = cgraph->nodes[node_idx + 1 + j];
        if (vw->op != GGML_OP_VIEW || vw->src[0] != mul || vw->ne[0] != n_embd || vw->ne[1] != nt) { return false; }
    }
    const ggml_tensor * final_add = cgraph->nodes[node_idx + n_nodes - 1];
    if (final_add->op != GGML_OP_ADD || final_add->type != GGML_TYPE_F32 ||
        final_add->ne[0] != n_embd || final_add->ne[1] != nt || final_add->ne[2] != 1) { return false; }
    if (!ggml_is_contiguous(final_add)) { return false; }
    // the fused kernel reads experts + writes final_add in one pass; bail if the
    // pool allocator overlapped the output with the (large) experts input -- would race.
    // The small weights input is copied to a private scratch in the dispatch, so its own
    // aliasing with the output is handled there and does not block the fusion.
    if (ggml_cl_tensors_overlap(experts, final_add)) { return false; }

    *out_final_add = final_add;
    return true;
}

static bool ggml_opencl_can_fuse_moe_bias_glu(const ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 4 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_op ops[] = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };
    const int outputs[] = { node_idx + 4 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, 5, ops, outputs, 1)) {
        return false;
    }

    const ggml_tensor * gate_mm  = cgraph->nodes[node_idx];
    const ggml_tensor * gate_add = cgraph->nodes[node_idx + 1];
    const ggml_tensor * up_mm    = cgraph->nodes[node_idx + 2];
    const ggml_tensor * up_add   = cgraph->nodes[node_idx + 3];
    const ggml_tensor * glu      = cgraph->nodes[node_idx + 4];

    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_OAI || gate_mm->src[1]->ne[2] == 1) {
        return false;
    }
    if (gate_add->src[0] != gate_mm || up_add->src[0] != up_mm ||
        glu->src[0] != gate_add || glu->src[1] != up_add ||
        up_mm->src[1] != gate_mm->src[1] || up_mm->src[2] != gate_mm->src[2]) {
        return false;
    }
    if (ggml_get_op_params_i32(glu, 1)) {
        return false;
    }
    if (gate_add->type != GGML_TYPE_F32 || up_add->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (!gate_add->src[1] || gate_add->src[1]->type != GGML_TYPE_F32 ||
        !up_add->src[1] || up_add->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (!gate_add->src[2] || gate_add->src[2]->type != GGML_TYPE_I32 || up_add->src[2] != gate_add->src[2]) {
        return false;
    }
    if (!ggml_are_same_shape(gate_add, up_add) || glu->ne[0] != gate_add->ne[0] ||
        glu->ne[1] != gate_add->ne[1] || glu->ne[2] != gate_add->ne[2] || glu->ne[3] != gate_add->ne[3]) {
        return false;
    }
    if (gate_add->ne[3] != 1) {
        return false;
    }
    return ggml_is_contiguous(glu) && ggml_is_contiguous(gate_mm) && ggml_is_contiguous(up_mm);
}

static bool ggml_opencl_can_fuse_moe_bias_combine(const ggml_cgraph * cgraph, int node_idx,
                                                  const ggml_tensor ** out_final_add) {
    if (node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[node_idx];
    if (add->op != GGML_OP_ADD_ID) {
        return false;
    }
    const ggml_tensor * mul = cgraph->nodes[node_idx + 1];
    if (mul->op != GGML_OP_MUL || mul->src[0] != add) {
        return false;
    }

    const ggml_tensor * final_add = NULL;
    if (!ggml_opencl_can_fuse_moe_combine(cgraph, node_idx + 1, &final_add)) {
        return false;
    }

    const ggml_tensor * raw  = add->src[0];
    const ggml_tensor * bias = add->src[1];
    const ggml_tensor * ids  = add->src[2];
    if (!raw || !bias || !ids) {
        return false;
    }
    if (raw->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || add->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_are_same_shape(raw, add) || !ggml_is_contiguous(raw)) {
        return false;
    }
    if (raw->nb[1] != add->nb[1] || raw->nb[2] != add->nb[2]) {
        return false;
    }
    if (ids->ne[0] < add->ne[1] || ids->ne[1] < add->ne[2]) {
        return false;
    }

    const int k = (int) add->ne[1];
    const int n_nodes = 2 + k + (k - 1);
    if (n_nodes >= 32 || node_idx + n_nodes > cgraph->n_nodes) {
        return false;
    }
    ggml_op ops[32];
    int n = 0;
    ops[n++] = GGML_OP_ADD_ID;
    ops[n++] = GGML_OP_MUL;
    for (int j = 0; j < k; ++j) {
        ops[n++] = GGML_OP_VIEW;
    }
    for (int j = 0; j < k - 1; ++j) {
        ops[n++] = GGML_OP_ADD;
    }
    const int outputs[] = { node_idx + n_nodes - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, n_nodes, ops, outputs, 1)) {
        return false;
    }

    *out_final_add = final_add;
    return true;
}

static bool ggml_opencl_can_fuse(const ggml_backend_opencl_context * backend_ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {

    // glu(mul_mat(Wg,x), mul_mat(Wu,x)) - the FFN gate/up GEMVs + GLU. This is a
    // non-linear subgraph (up does NOT consume gate), so the contiguous
    // ggml_can_fuse below rejects it; use ggml_can_fuse_subgraph with the glu as
    // the sole output and validate the edges explicitly. q4_K decode only;
    // byte-identical to the per-op path.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT &&
        ops.begin()[1] == GGML_OP_MUL_MAT && ops.begin()[2] == GGML_OP_GLU) {
        const enum ggml_op glu_ops[] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
        const int          glu_out[] = { node_idx + 2 };
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, 3, glu_ops, glu_out, 1)) {
            return false;
        }

        const ggml_tensor *gate = cgraph->nodes[node_idx];
        const ggml_tensor *up   = cgraph->nodes[node_idx+1];
        const ggml_tensor *glu  = cgraph->nodes[node_idx+2];

        // decode GEMV path only (single token); prefill GEMM is separate
        if (gate->ne[1] != 1 || up->ne[1] != 1) {
            return false;
        }
        // both projections must be q4_K weights, f32 activation/output
        if (gate->src[0]->type != GGML_TYPE_Q4_K || up->src[0]->type != GGML_TYPE_Q4_K ||
            gate->src[1]->type != GGML_TYPE_F32  || up->src[1]->type != GGML_TYPE_F32  ||
            gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
            return false;
        }
        // gate and up must share the same activation and have matching shape/stride
        if (gate->src[1] != up->src[1] ||
            !ggml_are_same_shape(gate->src[0], up->src[0]) ||
            !ggml_are_same_stride(gate->src[0], up->src[0])) {
            return false;
        }
        // GLU must read gate as src[0] and up as src[1], no swap (the fused
        // epilogue applies the activation to gate, multiplies by up)
        if (glu->src[0] != gate || glu->src[1] != up) {
            return false;
        }
        if (ggml_get_op_params_i32(glu, 1) /* swapped */) {
            return false;
        }
        // SWIGLU_OAI carries extra alpha/limit params -> not handled by the fused kernel
        if (ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU_OAI) {
            return false;
        }
        // the fused kernel reads the standard noshuffle image layout; the tiled
        // layout packs weights differently -> defer those to the per-op path
        if (use_q4k_tiled(backend_ctx, gate->src[0]) || use_q4k_tiled(backend_ctx, up->src[0])) {
            return false;
        }
        // q4_K bin kernel requires 32b transposed layout, not compatible with the fused gemv
        if (use_q4_k_bin_kernels(backend_ctx, gate->src[0]) || use_q4_k_bin_kernels(backend_ctx, up->src[0])) {
            return false;
        }
        // that noshuffle layout is only produced at set_tensor time when
        // use_adreno_kernels() accepts the weight (ne0 >= 512 && ne1 >= 512).
        // Smaller weights stay in the plain q4_K layout, which this kernel would
        // misread -> defer them to the per-op path. Real FFN gate/up weights are
        // far above the threshold, so production dispatch is unchanged.
        if (!use_adreno_kernels(backend_ctx, gate->src[0]) ||
            !use_adreno_kernels(backend_ctx, up->src[0])) {
            return false;
        }
        return true;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        // rms_norm only supports f32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms_norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] &&
            !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        // rms_norm assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        const ggml_tensor *norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul  = cgraph->nodes[node_idx+1];
        const ggml_tensor *add  = cgraph->nodes[node_idx+2];
        const ggml_tensor *w    = mul->src[0] == norm ? mul->src[1] : mul->src[0];
        const ggml_tensor *b    = add->src[0] == mul  ? add->src[1] : add->src[0];

        // norm fusion only supports F32
        if (norm->src[0]->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            return false;
        }

        if (norm->src[0]->ne[0] % 4 != 0) {
            return false;
        }

        if (!ggml_is_contiguous(norm->src[0]) || !ggml_is_contiguous(w) || !ggml_is_contiguous(b)) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        // rms_norm(x) * w + b, fused (residual). Mirrors the RMS_NORM+MUL gate
        // plus the residual-add operand's constraints.
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = cgraph->nodes[node_idx+2];
        const ggml_tensor *w        = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
        const ggml_tensor *b        = add->src[0] == mul      ? add->src[1] : add->src[0];

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        if (w->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            b->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
            return false;
        }
        if (rms_norm->src[0]->ne[0] % 4 != 0) {
            return false;
        }
        // if rms_norm is the B operand of mul, broadcast is not handled
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }
        // the residual must match the normed output shape (no add broadcast)
        if (!ggml_are_same_shape(b, add)) {
            return false;
        }
        // rms_norm assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1]) ||
            !ggml_is_contiguous_rows(b)) {
            return false;
        }
    } else if (ops.size() == 3 && ops.begin()[0] == GGML_OP_GROUP_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD) {
        const ggml_tensor *gn = cgraph->nodes[node_idx];
        const ggml_tensor *mul = cgraph->nodes[node_idx+1];
        const ggml_tensor *add = cgraph->nodes[node_idx+2];
        const ggml_tensor *w   = mul->src[0] == gn ? mul->src[1] : mul->src[0];
        const ggml_tensor *b   = add->src[0] == mul ? add->src[1] : add->src[0];

        if (gn->src[0]->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            return false;
        }

        if (!ggml_is_contiguous(gn->src[0]) || !ggml_is_contiguous(w) || !ggml_is_contiguous(b)) {
            return false;
        }
    }

    return true;
}

static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // NOTE: this may oversynchronize by synchronizing with
        //       backends/devices which don't compute 'cgraph's
        //       dependencies.
        sync_with_other_backends(backend);

        if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            ggml_opencl_op_norm_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            i += 2;
            continue;
        }
        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_GROUP_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            ggml_opencl_op_group_norm_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            i += 2;
            continue;
        }
        // Fuse the MoE combine: router-weight mul + cross-expert add chain ->
        // one weighted-sum-across-experts kernel.
        // Fold the gpt-oss MoE bias epilogue: add_id(gate_bias) + add_id(up_bias) +
        // glu(swiglu_oai) -> one kernel, leaving the two matmuls as their own dispatches.
        // Both add_ids are in-place passes over a tensor the GLU reads again, so this
        // drops two full read+write passes per layer. Opt out GGML_OPENCL_FUSE_MOE_BIAS_GLU=0.
        if (backend_ctx->fuse_moe_bias_glu && !backend_ctx->disable_fusion &&
            ggml_opencl_can_fuse_moe_bias_glu(cgraph, i)) {
            ggml_cl_moe_bias_glu_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2],
                                       cgraph->nodes[i+3], cgraph->nodes[i+4]);
            i += 4;
            continue;
        }

        // Fold the MoE down-projection bias into the combine: add_id(down_bias) + the whole
        // combine subgraph -> one kernel. Checked before the plain combine arm so the longer
        // pattern wins. Opt out GGML_OPENCL_FUSE_MOE_BIAS_COMBINE=0.
        if (backend_ctx->fuse_moe_bias_combine && backend_ctx->fuse_moe_combine &&
            !backend_ctx->disable_fusion) {
            const ggml_tensor * bias_combine_out = nullptr;
            if (ggml_opencl_can_fuse_moe_bias_combine(cgraph, i, &bias_combine_out)) {
                ggml_cl_moe_bias_combine_fused(backend, node, cgraph->nodes[i+1], bias_combine_out);
                i += 2 * (int)node->ne[1];   // ADD_ID + MUL + k VIEWs + (k-1) ADDs
                continue;
            }
        }

        if (backend_ctx->fuse_moe_combine && !backend_ctx->disable_fusion) {
            const ggml_tensor * combine_out = nullptr;
            if (ggml_opencl_can_fuse_moe_combine(cgraph, i, &combine_out)) {
                ggml_cl_moe_combine_fused(backend, node, combine_out);
                i += 2 * (int)node->ne[1] - 1;   // skip the k VIEWs + (k-1) ADDs
                continue;
            }
        }

        // Fuse rms_norm + mul(weight) + add(residual). Checked before the
        // rms_norm+mul fuse so the 3-op pattern wins over its 2-op prefix.
        // Default on, opt-out GGML_OPENCL_FUSE_RMS_ADD=0.
        if (!backend_ctx->disable_fusion && backend_ctx->fuse_rms_add &&
            ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD })) {
            ggml_opencl_op_rms_norm_mul_add_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            i += 2;
            continue;
        }
        if (!backend_ctx->disable_fusion && ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_opencl_op_rms_norm_fused(backend, node, cgraph->nodes[i+1]);
            i++;
            continue;
        }
        // Fuse mul_mat(Wg,x) + mul_mat(Wu,x) + glu - fold the FFN's two decode
        // GEMVs and the GLU into one dispatch. q4_K only (guarded below); the
        // fused kernel uses the same accumulation/reduction order and the same
        // scalar GLU formula -> coherent. Default on, opt-out GGML_OPENCL_FUSE_MM_GLU=0.
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
        // The fused executor (ggml_cl_mul_mat_q4_k_glu_fused) is image-path /
        // Adreno-only (GGML_ABORT on the non-Adreno #else); gate the dispatch to
        // match so the FFN GLU subgraph stays dormant on Intel/other drivers.
        if (backend_ctx->fuse_mm_glu && !backend_ctx->disable_fusion &&
            ggml_opencl_can_fuse(backend_ctx, cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU })) {
            ggml_cl_mul_mat_q4_K_glu_fused(backend, node, cgraph->nodes[i+1], cgraph->nodes[i+2]);
            i += 2;
            continue;
        }
#endif

        bool ok = ggml_cl_compute_forward(backend, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);
    }

    return GGML_STATUS_SUCCESS;
}

// The optimized gemm and gemv kernels are used for large matrices without batch.
// tensor is the quantized weights matrix.
static bool ggml_opencl_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_opencl_device_context * dev_ctx     = (ggml_backend_opencl_device_context *)dev->context;
    ggml_backend_opencl_context *        backend_ctx = dev_ctx->backend_ctx;

    // reject ops that match the opfilter regex
    if (dev_ctx->opfilter && std::regex_match(std::string(ggml_op_desc(op)), *dev_ctx->opfilter)) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
            return true;
        case GGML_OP_GET_ROWS:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                    return true;
                case GGML_TYPE_Q4_0:
#ifdef GGML_OPENCL_SOA_Q
                    // We do not support flattened Q4_0 (and possibly other Q's)
                    return false;
#else // GGML_OPENCL_SOA_Q
                    return true;
#endif // GGML_OPENCL_SOA_Q
                default:
                    return false;
            }
        case GGML_OP_SET_ROWS:
            {
                // TODO: add support
                // ref: https://github.com/ggml-org/llama.cpp/pull/14274
#pragma message("TODO: implement BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL support (https://github.com/ggml-org/llama.cpp/pull/14661)")
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }
                switch (op->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                        return (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                    default:
                        return false;
                }
            }
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                case GGML_TYPE_F16:
                    switch (op->type) {
                        case GGML_TYPE_F16:
                        case GGML_TYPE_F32:
                            return true;
                        default:
                            return false;
                    }
                case GGML_TYPE_I32:
                    switch (op->type) {
                        case GGML_TYPE_I32:
                            return true;
                        default:
                            return false;
                    }
                default:
                    return false;
            }
        case GGML_OP_SET: {
            return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32) &&
                    op->type == op->src[0]->type &&
                    op->type == op->src[1]->type;
        }
        case GGML_OP_SCALE:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_ADD:
            if (op->type == GGML_TYPE_F16) {
                const bool src0_ok = op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32;
                const bool src1_ok = op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32;
                if (src0_ok && src1_ok) {
                    return true;
                }
            }
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
            return (op->src[0]->type == op->src[1]->type) &&
                   (op->src[0]->type == op->type) &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_ADD_ID:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                    ggml_is_contiguous(op->src[0]);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                    return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
                case GGML_UNARY_OP_SIGMOID:
                    return ggml_is_contiguous(op->src[0]);
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_EXP:
                    // Adreno F16 exp/expm1 overflow even post-half->float convert.
                    return op->src[0]->type == GGML_TYPE_F32;
                case GGML_UNARY_OP_EXPM1:
                    return op->src[0]->type == GGML_TYPE_F32;
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
                case GGML_UNARY_OP_SOFTPLUS:
                    return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_TRI:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_FILL:
            return op->type == GGML_TYPE_F32 && ggml_is_contiguous(op);
        case GGML_OP_CLAMP:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SOFT_MAX:
        case GGML_OP_NORM:
            return true;
        case GGML_OP_RMS_NORM:
            return op->ne[0] % 4 == 0 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_REPEAT:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32; // Assuming F32 for now, can be expanded
        case GGML_OP_PAD:
            // TODO: add circular padding support for opencl, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_UPSCALE: {
            ggml_scale_mode mode = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & 0xFF);
            const bool antialias = (ggml_scale_mode)(ggml_get_op_params_i32(op, 0) & GGML_SCALE_FLAG_ANTIALIAS);
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   (mode == GGML_SCALE_MODE_NEAREST || mode == GGML_SCALE_MODE_BILINEAR) && !antialias;
        }
        case GGML_OP_CONV_2D:
            return (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16) ||
                   (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
                   (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        case GGML_OP_SSM_CONV:
            return (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
        case GGML_OP_SSM_SCAN: {
                if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32 || op->src[2]->type != GGML_TYPE_F32 ||
                    op->src[3]->type != GGML_TYPE_F32 || op->src[4]->type != GGML_TYPE_F32 ||
                    op->src[5]->type != GGML_TYPE_F32 || op->src[6]->type != GGML_TYPE_I32) {
                    return false;
                }

                const int64_t d_state = op->src[0]->ne[0];
                return d_state >= 1 && d_state <= 256 && (d_state & (d_state - 1)) == 0;
            }
        case GGML_OP_GATED_DELTA_NET:
            {
                // Match the Vulkan backend: only F32 -> F32, S_v in {16, 32, 64, 128}.
                if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
                    return false;
                }
                const int64_t S_v = op->src[2]->ne[0];
                return S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128;
            }
        case GGML_OP_CONCAT:
            {
                const ggml_type t = op->src[0]->type;
                return op->src[1]->type == t && op->type == t &&
                       !ggml_is_quantized(t) && ggml_blck_size(t) == 1 &&
                       (ggml_type_size(t) == 1 || ggml_type_size(t) == 2 ||
                        ggml_type_size(t) == 4 || ggml_type_size(t) == 8);
            }
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
            if (op->src[0]->type == GGML_TYPE_F16) {
                return true;
            } else if (op->src[0]->type == GGML_TYPE_BF16) {
                return true;
            } else if (op->src[0]->type == GGML_TYPE_F32) {
                return op->src[1]->type == GGML_TYPE_F32;
            } else if (op->src[0]->type == GGML_TYPE_Q1_0) {
                return op->src[1]->type == GGML_TYPE_F32;
            } else if (op->src[0]->type == GGML_TYPE_Q4_0) {
                // Non-contig src0 routes through on-device dequant-to-f16.
                return op->src[1]->type == GGML_TYPE_F32;
            } else if (op->src[0]->type == GGML_TYPE_Q4_1 ||
                       op->src[0]->type == GGML_TYPE_Q5_0  || op->src[0]->type == GGML_TYPE_Q5_1 ||
                       op->src[0]->type == GGML_TYPE_MXFP4 ||
                       op->src[0]->type == GGML_TYPE_IQ4_NL ||
                       op->src[0]->type == GGML_TYPE_Q4_K  ||
                       op->src[0]->type == GGML_TYPE_Q5_K  ||
                       op->src[0]->type == GGML_TYPE_Q6_K) {
                // The E031.41 compiler (usually with A7x) miscompiles the flat K-quant
                // GEMV kernels (kernel_mul_mv_q*_K_f32_flat) and makes lm_head run much
                // slower than it should. So, make it fallback to CPU to preserve performance
                // for this compiler series.
                static const char * a7x_lmhead_env = getenv("GGML_OPENCL_A7X_LMHEAD_CPU");
                static const bool   a7x_lmhead_cpu = (a7x_lmhead_env == nullptr || a7x_lmhead_env[0] != '0');
                if (a7x_lmhead_cpu &&
                    backend_ctx->adreno_gen == ADRENO_GPU_GEN::A7X &&
                    (op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q5_K ||
                     op->src[0]->type == GGML_TYPE_Q6_K) &&
                    op->src[0]->ne[1] >= 32768) {   // vocab-scale weight; no FFN/attn weight is this tall
                    return false;
                }
                // The generic mul_mv (GEMV) kernels are wrong for large-batch prefill on
                // Adreno. A quant mul_mat only avoids the GEMV when it reaches the Adreno
                // trans-weight GEMM, which needs both a GEMM kernel for the type and
                // use_adreno_kernels(). Decline the large-N shapes that would otherwise
                // fall through to the GEMV.
                {
                    const ggml_type t = op->src[0]->type;
                    const bool type_has_gemm = (t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
                                                t == GGML_TYPE_IQ4_NL || t == GGML_TYPE_Q8_0 ||
                                                t == GGML_TYPE_Q4_K  || t == GGML_TYPE_Q5_K  ||
                                                t == GGML_TYPE_Q6_K);
                    const bool uses_gemm = type_has_gemm && use_adreno_kernels(backend_ctx, op->src[0]);
                    if (!uses_gemm && op->src[1]->ne[1] >= 512) {
                        return false;
                    }
                }
                return op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            } else if (op->src[0]->type == GGML_TYPE_Q8_0) {
                // ggml_cl_mul_mat_q8_0_f32_adreno now honors src1/dst view_offs (the
                // activation sub-buffer starts at offset1 and the kernels take offsetd),
                // so a broadcast q8_0 matmul (src1 batch > src0 batch, e.g. Qwen3.5-9B-UD
                // / Qwen3.6-35B q8_0 GDN ssm_out) runs on GPU via the per-slice broadcast
                // iteration in ggml_cl_mul_mat. No special-casing needed.
                return op->src[1]->type == GGML_TYPE_F32;
            }
            return false;
        case GGML_OP_MUL_MAT_ID:
            if (op->src[0]->type == GGML_TYPE_Q4_0 ||
                op->src[0]->type == GGML_TYPE_Q8_0 ||
                op->src[0]->type == GGML_TYPE_MXFP4) {
                if (op->src[1]->type == GGML_TYPE_F32) {
                    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                }
            }
            // q4_0, q8_0 and mxfp4 have general MUL_MAT_ID support,
            // the quantizations here currently do not - they are only supported by Adreno with certain shapes
            if (op->src[0]->type == GGML_TYPE_Q4_1 ||
                op->src[0]->type == GGML_TYPE_Q5_0 ||
                op->src[0]->type == GGML_TYPE_Q5_1 ||
                op->src[0]->type == GGML_TYPE_Q4_K ||
                op->src[0]->type == GGML_TYPE_Q5_K ||
                op->src[0]->type == GGML_TYPE_Q6_K) {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
                if (op->src[1]->type == GGML_TYPE_F32) {
                    return use_adreno_moe_kernels(backend_ctx, op->src[0])
                        && ggml_is_contiguous(op->src[0])
                        && ggml_is_contiguous(op->src[1]);
                }
#endif
                return false;
            }
            return false;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return op->ne[3] == 1;
        case GGML_OP_ROPE: {
            const int mode = ((const int32_t *) op->op_params)[2];
            const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
            const bool is_vision = mode == GGML_ROPE_TYPE_VISION;
            if (is_mrope && !is_vision) {
                if (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            }
            if (is_vision) {
                if (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            }
            return true;
        }
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_IM2COL:
            return true;
        case GGML_OP_ARGSORT: {
            ggml_cl_load_kernels_argsort(backend_ctx);

            cl_kernel kernel = backend_ctx->argsort.kernel_argsort_f32_i32;
            int max_workgroup_size = backend_ctx->get_kernel_workgroup_size(kernel);

            int cols = 1;
            while (cols < op->ne[0]) {
                cols *= 2;
            }

            return cols <= max_workgroup_size && op->src[0]->type == GGML_TYPE_F32;
        }
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
            return op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT: {
#ifdef GGML_OPENCL_USE_ADRENO_KERNELS
            if (use_fa_bin_kernels_prefill(backend_ctx, op->src[0], op->src[1], op->src[2])) {
                return true;
            }
#endif
            // The E17 compilers segfault while building FA kernels, skip E17 for now
            if (adreno_e17_compiler_quirks(backend_ctx)) {
                return false;
            }
            const ggml_tensor * q = op->src[0];
            const ggml_tensor * k = op->src[1];
            const ggml_tensor * v = op->src[2];

            const int dk = q->ne[0];
            const int dv = v->ne[0];

            const struct { int dk; int dv; } supported_dims[] = {
                { 40,  40}, { 64,  64}, { 80,  80}, { 96,  96},
                {112, 112}, {128, 128}, {192, 128},
                {192, 192}, {256, 256},
                {512, 512},
            };

            bool dims_supported = false;
            for (size_t i = 0; i < sizeof(supported_dims)/sizeof(supported_dims[0]); ++i) {
                if (supported_dims[i].dk == dk && supported_dims[i].dv == dv) {
                    dims_supported = true;
                    break;
                }
            }
            if (!dims_supported) {
                return false;
            }

            const bool is_f32_f32  = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 &&
                                     v->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
            const bool is_f16_f16  = q->type == GGML_TYPE_F16 && k->type == GGML_TYPE_F16 &&
                                     v->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16;
            const bool is_f32_f16  = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 &&
                                     v->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F32;

            const bool is_f32_q8_0 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_Q8_0 &&
                                     v->type == GGML_TYPE_Q8_0 && op->type == GGML_TYPE_F32 &&
                                     dk % 32 == 0 && dv % 32 == 0;
            const bool is_f32_q4_0 = q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_Q4_0 &&
                                     v->type == GGML_TYPE_Q4_0 && op->type == GGML_TYPE_F32 &&
                                     dk % 32 == 0 && dv % 32 == 0;

            // A7X (Adreno 740, compiler E031.41) SIGSEGVs inside clBuildProgram
            // building the flash_attn programs whose KV path is mixed-type or
            // dequantized - f32_f16, q8_0, q4_0 (reproduced at DK=40 and DK=64; it
            // is DK-independent). It is a driver crash, not codegen-wrong-output, so
            // it cannot be caught in-process (fatal=false only handles clean compile
            // errors). The uniform f16_f16 / f32_f32 programs compile fine on this
            // compiler, so decline only the KV-convert variants; ggml then runs
            // those (f16-KV / quant-KV) attention layers on the CPU backend.
            // Negative compiler carve-out, same idiom as the Intel DK=512 decline
            // below and the X1E driver-quirk guards.
            if (backend_ctx && backend_ctx->adreno_gen == ADRENO_GPU_GEN::A7X &&
                (is_f32_f16 || is_f32_q8_0 || is_f32_q4_0)) {
                return false;
            }

            // Asymmetric KV: host-dequants both sides to F32, uses f32 kernel.
            auto is_kv_type_ok = [](ggml_type t) {
                return t == GGML_TYPE_F16 || t == GGML_TYPE_F32 ||
                       t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q8_0;
            };
            const bool is_f32_asym = q->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                                     k->type != v->type &&
                                     is_kv_type_ok(k->type) && is_kv_type_ok(v->type);

            const bool kv_combo_ok = is_f32_f32 || is_f16_f16 || is_f32_f16 ||
                                         is_f32_q8_0 || is_f32_q4_0 || is_f32_asym;
            if (!kv_combo_ok) {
                return false;
            }

            // Some compilers for A7x (Adreno 740, compiler E031.41) crashes when
            // building FA kernels with mixed or quant types (f32_f16, f32_q8_0, f32_q4_0)
            // Here we skip all A7x for these kernels to avoid crash
            if (backend_ctx->adreno_gen == ADRENO_GPU_GEN::A7X &&
                (is_f32_f16 || is_f32_q8_0 || is_f32_q4_0)) {
                return false;
            }

            if (dk == 512) {
                if (backend_ctx->gpu_family == INTEL) {
                    return false;
                }
                if (!is_f32_f16) {
                    return false;
                }
                if (q->ne[1] == 1) {
                    // DK=512 decode is bandwidth-bound and slower on the GPU
                    // than on the CPU; decline it here so it runs on the CPU.
                    // Prefill (n_q > 1) stays on the GPU.
                    return false;
                } else {
                    // prefill, BM-tile in its own FA_PREFILL_ONLY program
                    if (!ggml_opencl_ensure_fa_f32_f16_prefill_512(backend_ctx, /*split=*/false)) {
                        return false;
                    }
                }
            }
            return true;
        }
        default:
            return false;
    }
}

// Forward declaration - implementation appears later in the file.
static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type);

static ggml_guid_t ggml_backend_opencl_guid() {
    static ggml_guid guid = { 0xde, 0xe0, 0x70, 0xa2, 0x73, 0x4e, 0x4d, 0xbc, 0xb0, 0xc7, 0x4f, 0xd4, 0x6d, 0x4e, 0x90, 0xfe };
    return &guid;
}

static ggml_backend_i ggml_backend_opencl_i = {
    /* .get_name                = */ ggml_backend_opencl_name,
    /* .free                    = */ ggml_backend_opencl_free,
    /* .set_tensor_async        = */ NULL,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ ggml_backend_opencl_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_opencl_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_t ggml_backend_opencl_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_opencl_reg(), 0);
    ggml_backend_opencl_context *backend_ctx = ggml_cl_init(dev);
    backend_ctx->ref_count++;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_opencl_guid(),
        /* .iface   = */ ggml_backend_opencl_i,
        /* .device  = */ dev,
        /* .context = */ backend_ctx
    };

    return backend;
}

bool ggml_backend_is_opencl(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_opencl_name;
}

//
// buffer
//

static void ggml_backend_opencl_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_opencl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer->buft->device->context;
    return (void *) (uintptr_t) dev_ctx->backend_ctx->alignment;
}

static enum ggml_status ggml_backend_opencl_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);

        ggml_tensor_extra_cl * view_extra = (ggml_tensor_extra_cl *) tensor->view_src->extra;
        GGML_ASSERT(view_extra && "view_extra is nullptr?");

        // Reuse extra of the parent tensor. The offset of this view tensor
        // becomes `extra->offset + view_offs` and needs to be calculated when
        // it is used. This changes is needed because of the change to
        // ggml_alloc.c in https://github.com/ggml-org/llama.cpp/pull/7640.
        // `buffer` passed in here will always be `tensor->buffer`. It is OK
        // to allocate extras from the same buffer context for ordinary
        // intermediate tensors. But for views into kv cache tensors, doing so
        // would mess up the extras used by kv cache.
        // Before #7640, `buffer` is for intermediate tensors, which is always
        // different from that of kv cache tensors.
        //
        // NB: now extra->offset no longer accounts for view_offs.
        // NB: this should not apply to weight tensors (for end-to-end runs, but
        //     may apply for test-backend-ops).
        // FIXME: if any unexpected results are seen, double check the offset -
        // there could be other places that need fix.
        tensor->extra = view_extra;
    } else {
        {
            size_t offset = (char *) tensor->data - (char *) ggml_backend_opencl_buffer_get_base(buffer);

            ggml_tensor_extra_cl * extra = ctx->ggml_opencl_alloc_temp_tensor_extra();
            extra->offset = offset;
            extra->data_device = ctx->buffer[0];
            extra->actual_size = ggml_nbytes(tensor);

            tensor->extra = extra;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_opencl_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer->buft->device->context;
    ggml_backend_opencl_context * backend_ctx = dev_ctx->backend_ctx;

    cl_command_queue queue = backend_ctx->queue;

    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    for (cl_mem buf : ctx->buffer) {
        CL_CHECK(clEnqueueFillBuffer(queue, buf, &value, sizeof(value), 0, buffer->size, 0, NULL, NULL));
    }
    CL_CHECK(clFinish(queue));
}

static void ggml_backend_opencl_buffer_reset(ggml_backend_buffer_t buffer) {
    ggml_backend_opencl_buffer_context * ctx = (ggml_backend_opencl_buffer_context *) buffer->context;
    ctx->reset();
}

static ggml_backend_buffer_i ggml_backend_opencl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_opencl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_opencl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_opencl_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_opencl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_opencl_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_opencl_buffer_clear,
    /* .reset           = */ ggml_backend_opencl_buffer_reset,
};

//
// buffer type
//

static const char * ggml_backend_opencl_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "OpenCL";

    GGML_UNUSED(buffer_type);
}

static ggml_backend_buffer_t ggml_backend_opencl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buffer_type, size_t size) {
    ggml_backend_opencl_context *backend_ctx = ggml_cl_init(buffer_type->device);

    if (!backend_ctx->program_cache_initialized) {
        backend_ctx->program_cache = cl_program_cache_init(backend_ctx->device);
        backend_ctx->program_cache_initialized = true;
    }
    load_cl_kernels(backend_ctx);

    // clCreateBuffer returns -61 for size 0
    size = std::max(size, (size_t)1);

    cl_int err;
    cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    // On Adreno X1-85 the device pool intermittently fails at hundreds of MB
    // once the heap fragments (e.g. graph-allocator compute-buffer reserve
    // after model load). Four-step retry:
    //   1. normal alloc (fast path)
    //   2. clFinish + retry (drains in-flight allocs)
    //   3. cl_qcom_large_buffer (X2-class driver only, OpenCL 3.0 only)
    //   4. ALLOC_HOST_PTR (host-pinned pool) - last-resort fallback. This
    //      buffer backs compute scratch read/written by every kernel in the
    //      graph, so kernel accesses fall to host memory and runtime perf
    //      degrades meaningfully. Better than failing to load, but the user
    //      should see the warning and consider -ngl reduction.
    if (err != CL_SUCCESS) {
        clFinish(backend_ctx->queue);
        mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE, size, NULL, &err);
    }
#if GGML_OPENCL_TARGET_VERSION >= 300
    // clCreateBufferWithProperties and cl_mem_properties are OpenCL 3.0. Drivers older than
    // that do not export the symbol, so a build targeting them fails to link. The large
    // buffer extension is only ever enabled on drivers that are well past 3.0, so this path
    // is dead there anyway.
    if (err != CL_SUCCESS && backend_ctx->adreno_use_large_buffer) {
        cl_mem_properties props[] = { 0x41A6 /* CL_LARGE_BUFFER_QCOM */, 1, 0 };
        mem = clCreateBufferWithProperties(backend_ctx->context, props, CL_MEM_READ_WRITE, size, NULL, &err);
    }
#endif
    if (err != CL_SUCCESS) {
        mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);
        if (err == CL_SUCCESS) {
            GGML_LOG_WARN("%s: %.2f MiB allocated via CL_MEM_ALLOC_HOST_PTR fallback - "
                          "device pool exhausted; runtime perf will be degraded. "
                          "Consider lowering -ngl or context size.\n",
                          __func__, size / 1024.0 / 1024.0);
        }
    }

    if (err != CL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to allocate %.2f MiB (err=%d). "
                       "Consider reducing -ngl, lowering -c / -ub, or using quantized KV cache.\n",
                       __func__, size / 1024.0 / 1024.0, err);
        return nullptr;
    }

    ggml_backend_opencl_buffer_context * ctx = new ggml_backend_opencl_buffer_context(mem);

    return ggml_backend_buffer_init(buffer_type, ggml_backend_opencl_buffer_interface, ctx, size);
}

static size_t ggml_backend_opencl_buffer_type_get_alignment(ggml_backend_buffer_type_t buffer_type) {
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer_type->device->context;
    return dev_ctx->backend_ctx->alignment;
}

static size_t ggml_backend_opencl_buffer_type_get_max_size(ggml_backend_buffer_type_t buffer_type) {
    static size_t max_size = -1;
    if (max_size == (size_t)-1) {
        ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buffer_type->device->context;
        max_size = dev_ctx->backend_ctx->max_alloc_size;
    }
    return max_size;
}

static bool ggml_backend_opencl_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_opencl(backend);

    UNUSED(buft);
}

static size_t ggml_backend_opencl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
#ifdef GGML_OPENCL_SOA_Q
    // set_tensor carves quantized weights into per-component subbuffers (d/q,
    // ql/qh/s/d, ...) whose origins are each rounded up to the device base
    // alignment. When a component's size is not a multiple of the alignment
    // (e.g. q6_K [1536,49155]: size_s = 49155*96 leaves a 96-byte gap at 128-byte
    // alignment), the aligned carve extends past ggml_nbytes and the last
    // subbuffer would overlap the next tensor in the pool. Reserve the worst-case
    // carve slack: at most 5 components (q5_K), i.e. 4 aligned gaps.
    if (ggml_is_quantized(tensor->type)) {
        ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) buft->device->context;
        size += 4 * dev_ctx->backend_ctx->alignment;
    }
#endif // GGML_OPENCL_SOA_Q
    return size;
}

static ggml_backend_buffer_type_i ggml_backend_opencl_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_opencl_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_opencl_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_opencl_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_opencl_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_opencl_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

//
// backend device
//

static const char * ggml_backend_opencl_device_get_name(ggml_backend_dev_t dev) {
    return "GPUOpenCL";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_opencl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_opencl_device_context *dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    return dev_ctx->device_name.c_str();
}

static void ggml_backend_opencl_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;

    static const size_t opencl_extra_margin = 1024ull*1024ull*1024ull;

    // OpenCL does not provide reliable currently-free device memory.
    // Use total/global memory as a best-effort upper bound.
    // Improved safety: Reduce by a 1GiB extra margin for common --fit
    *total = dev_ctx->global_mem_size;
    *free  = *total > opencl_extra_margin ? *total - opencl_extra_margin : 0;
}

static enum ggml_backend_dev_type ggml_backend_opencl_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_opencl_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_opencl_device_get_name(dev);
    props->description = ggml_backend_opencl_device_get_description(dev);
    props->type        = ggml_backend_opencl_device_get_type(dev);
    ggml_backend_opencl_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = ggml_backend_dev_caps {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
        /* .mmap_support          = */ false,
    };
}

static ggml_backend_t ggml_backend_opencl_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_opencl_context * backend_ctx = ggml_cl_init(dev);
    // Getting a new reference to the backend, increase ref_count
    backend_ctx->ref_count++;

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */ ggml_backend_opencl_guid(),
        /* .interface = */ ggml_backend_opencl_i,
        /* .device    = */ dev,
        /* .context   = */ backend_ctx,
    };

    ggml_backend_opencl_device_context * dev_ctx = (ggml_backend_opencl_device_context *) dev->context;
    ggml_opencl_print_backend_info(dev_ctx);
    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(dev->context);

    dev_ctx->buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_backend_opencl_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ nullptr,
    };

    return &dev_ctx->buffer_type;
}

static ggml_backend_buffer_t ggml_backend_opencl_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    return nullptr;
}

static bool ggml_backend_opencl_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_cl_init(dev);
    return ggml_opencl_supports_op(dev, op);
}

static bool ggml_backend_opencl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Check 'dev' and 'buffer_type' are not objects belonging to this backend.
    if (dev->iface.get_name != ggml_backend_opencl_device_get_name ||
        buft->iface.get_name != ggml_backend_opencl_buffer_type_get_name) {
        return false;
    }

    // Check cl_context is the same. clEnqueue* commands may not use
    // buffers from another cl_context.
    ggml_backend_opencl_context * backend_ctx0 = ggml_cl_init(dev);
    ggml_backend_opencl_context * backend_ctx1 = ggml_cl_init(buft->device);
    return backend_ctx0->context == backend_ctx1->context;
}

struct ggml_backend_device_i ggml_backend_opencl_device_i = {
    /* .get_name             = */ ggml_backend_opencl_device_get_name,
    /* .get_description      = */ ggml_backend_opencl_device_get_description,
    /* .get_memory           = */ ggml_backend_opencl_device_get_memory,
    /* .get_type             = */ ggml_backend_opencl_device_get_type,
    /* .get_props            = */ ggml_backend_opencl_device_get_props,
    /* .init_backend         = */ ggml_backend_opencl_device_init,
    /* .get_buffer_type      = */ ggml_backend_opencl_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_opencl_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_opencl_device_supports_op,
    /* .supports_buft        = */ ggml_backend_opencl_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// Backend registry

static const char * ggml_backend_opencl_reg_get_name(ggml_backend_reg_t reg) {
    return "OpenCL";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_opencl_reg_device_count(ggml_backend_reg_t reg) {
    return g_ggml_backend_opencl_devices.size();

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_opencl_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < ggml_backend_opencl_reg_device_count(reg));

    return &g_ggml_backend_opencl_devices[index];

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_opencl_reg_i = {
    /* .get_name         = */ ggml_backend_opencl_reg_get_name,
    /* .device_count     = */ ggml_backend_opencl_reg_device_count,
    /* .device_get       = */ ggml_backend_opencl_reg_device_get,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_opencl_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;
    std::lock_guard<std::mutex> lock(mutex);

    if (initialized) {
        return &reg;
    }
    initialized = true;

    g_ggml_backend_opencl_devices = ggml_opencl_probe_devices(&reg);

    reg = ggml_backend_reg{
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_opencl_reg_i,
        /* .context     = */ NULL,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_opencl_reg)

//------------------------------------------------------------------------------
// Debugging utils
//------------------------------------------------------------------------------
#if 0
#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2,
    "wrong q4_0 block size/padding");

#define QK_MXFP4 32

#include <math.h>
#ifdef __cplusplus
#include "half.hpp"
#endif

static void dump_tensor(ggml_backend_t backend, const struct ggml_tensor * tensor) {
    void * buf = malloc(ggml_nbytes(tensor));

    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;
    cl_command_queue queue = backend_ctx->queue;
#ifdef GGML_OPENCL_SOA_Q
    void * buf_q;
    void * buf_d;
#endif

    // Make sure everything is done.
    CL_CHECK(clFinish(queue));

#ifdef GGML_OPENCL_SOA_Q
    if (tensor->type == GGML_TYPE_Q4_0) {
        ggml_tensor_extra_cl_q4_0 * extra = (ggml_tensor_extra_cl_q4_0 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK4_0 * QK4_0/2;
        size_t size_d = ggml_nelements(tensor)/QK4_0 * sizeof(ggml_fp16_t);
        GGML_ASSERT(size_q + size_d == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_d);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->d, CL_TRUE, 0, size_d, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else if (tensor->type == GGML_TYPE_MXFP4) {
        ggml_tensor_extra_cl_mxfp4 * extra = (ggml_tensor_extra_cl_mxfp4 *) tensor->extra;
        GGML_ASSERT(extra);

        size_t size_q = ggml_nelements(tensor)/QK_MXFP4 * QK_MXFP4/2;
        size_t size_e = ggml_nelements(tensor)/QK_MXFP4 * sizeof(char);
        GGML_ASSERT(size_q + size_e == ggml_nbytes(tensor));
        buf_q = malloc(size_q);
        buf_d = malloc(size_e);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->q, CL_TRUE, 0, size_q, buf_q, 0, NULL, NULL));
        CL_CHECK(clEnqueueReadBuffer(queue, extra->e, CL_TRUE, 0, size_e, buf_d, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    } else {
        // Read out the tensor from GPU memory.
        ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
        GGML_ASSERT(extra);

        CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
        CL_CHECK(clFinish(queue));
    }
#else
    // Read out the tensor from GPU memory.
    ggml_tensor_extra_cl * extra = (ggml_tensor_extra_cl *) tensor->extra;
    GGML_ASSERT(extra);

    CL_CHECK(clEnqueueReadBuffer(queue, extra->data_device, CL_TRUE,
        extra->offset, ggml_nbytes(tensor), buf, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));
#endif // GGML_OPENCL_SOA_Q

    // Open file and dump.
    char fname[512];
    snprintf(fname, sizeof(fname), "./tensor-dumps/%s.txt", tensor->name);
    FILE * f = fopen(fname, "w");
    if (!f) {
        printf("Failed to open %s\n", fname);
        return;
    }

    if (tensor->type == GGML_TYPE_F32) {
        float * data = (float *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_I32) {
        int * data = (int *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%d\n", data[i]);
        }
    } else if (tensor->type == GGML_TYPE_F16) {
#ifdef __cplusplus
        half_float::half * data = (half_float::half *) buf;
        for (int i = 0; i < ggml_nelements(tensor); ++i) {
            if (std::isnan(data[i])) {
                printf("NaN found: %s\n", tensor->name);
                break;
            }
            fprintf(f, "%f\n", float(data[i]));
        }
#endif
    } else if (tensor->type == GGML_TYPE_Q4_0) {
#ifdef GGML_OPENCL_SOA_Q
        ggml_fp16_t * data_d = (ggml_fp16_t *)buf_d;
        unsigned char * data_q = (unsigned char *)buf_q;

        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data_d[i]);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data_q[k]);
            }
            fprintf(f, "\n");
            data_q += QK4_0/2;
        }
        free(buf_d);
        free(buf_q);
#else
        block_q4_0 * data = (block_q4_0 *) buf;
        for (int i = 0; i < ggml_nelements(tensor)/QK4_0; ++i) {
            fprintf(f, "%04x, ", data[i].d);
            for (int k = 0; k < QK4_0/2; ++k) {
                fprintf(f, "%02x, ", data[i].qs[k]);
            }
            fprintf(f, "\n");
        }
#endif // GGML_OPENCL_SOA_Q
    }
    free(buf);
    fflush(f);
    fclose(f);
}
#else
#define dump_tensor(tensor)
#endif

//------------------------------------------------------------------------------
// Ops
//------------------------------------------------------------------------------

static bool ggml_cl_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst) {
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) &&
            src1->type == GGML_TYPE_F32 &&
             dst->type == GGML_TYPE_F32 &&
            (ne0 >= 32 && ne1 >= 32 && ne10 >= 32);
}

//------------------------------------------------------------------------------
// Op offloading
//------------------------------------------------------------------------------

typedef void (*ggml_cl_func_t)(ggml_backend_t backend, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

bool ggml_cl_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    ggml_cl_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    const bool any_on_device = tensor->extra
        || (src0 != nullptr && src0->extra)
        || (src1 != nullptr && src1->extra);

    switch (tensor->op) {
        case GGML_OP_GET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_get_rows;
            break;
        case GGML_OP_SET_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_set_rows;
            break;
        case GGML_OP_CPY:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_cpy;
            break;
        case GGML_OP_SET:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_set;
            break;
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_dup;
            break;
        case GGML_OP_ADD:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_add;
            break;
        case GGML_OP_ADD_ID:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_add_id;
            break;
        case GGML_OP_MUL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul;
            break;
        case GGML_OP_DIV:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_div;
            break;
        case GGML_OP_SUB:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sub;
            break;
        case GGML_OP_SQR:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sqr;
            break;
        case GGML_OP_SQRT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sqrt;
            break;
        case GGML_OP_MEAN:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mean;
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu;
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu_erf;
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_gelu_quick;
                    break;
                case GGML_UNARY_OP_SILU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_silu;
                    break;
                case GGML_UNARY_OP_RELU:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_relu;
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_sigmoid;
                    break;
                case GGML_UNARY_OP_TANH:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_tanh;
                    break;
                case GGML_UNARY_OP_NEG:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_neg;
                    break;
                case GGML_UNARY_OP_EXP:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_exp;
                    break;
                case GGML_UNARY_OP_EXPM1:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_expm1;
                    break;
                case GGML_UNARY_OP_ABS:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_abs;
                    break;
                case GGML_UNARY_OP_SGN:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_sgn;
                    break;
                case GGML_UNARY_OP_STEP:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_step;
                    break;
                case GGML_UNARY_OP_ELU:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_elu;
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_hardswish;
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_hardsigmoid;
                    break;
                case GGML_UNARY_OP_FLOOR:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_floor;
                    break;
                case GGML_UNARY_OP_CEIL:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_ceil;
                    break;
                case GGML_UNARY_OP_ROUND:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_round;
                    break;
                case GGML_UNARY_OP_TRUNC:
                    if (!any_on_device) { return false; }
                    func = ggml_cl_trunc;
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    if (!any_on_device) {
                        return false;
                    }
                    func = ggml_cl_softplus;
                    break;
                default:
                    return false;
            } break;
        case GGML_OP_GLU:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_glu;
            break;
        case GGML_OP_TRI:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_tri;
            break;
        case GGML_OP_FILL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_fill;
            break;
        case GGML_OP_CLAMP:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_clamp;
            break;
        case GGML_OP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_norm;
            break;
        case GGML_OP_RMS_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rms_norm;
            break;
        case GGML_OP_L2_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_l2_norm;
            break;
        case GGML_OP_GROUP_NORM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_group_norm;
            break;
        case GGML_OP_REPEAT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_repeat;
            break;
        case GGML_OP_PAD:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_pad(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_UPSCALE:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_upscale(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_CONV_2D:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_conv_2d;
            break;
        case GGML_OP_SSM_CONV:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_ssm_conv;
            break;
        case GGML_OP_SSM_SCAN:
            if (!any_on_device) {
                return false;
            }
            // SSM_SCAN has 7 source tensors, so it cannot use the standard
            // (src0, src1, dst) func signature. Dispatch directly and return.
            ggml_cl_ssm_scan(backend, tensor);
            return true;
        case GGML_OP_GATED_DELTA_NET:
            if (!any_on_device) {
                return false;
            }
            // GDN has 6 source tensors, so it cannot use the standard
            // (src0, src1, dst) func signature. Dispatch directly and return.
            ggml_cl_gated_delta_net(backend, tensor);
            return true;
        case GGML_OP_CONCAT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_concat;
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_timestep_embedding(backend, tensor->src[0], tensor);
            return true;
        case GGML_OP_MUL_MAT:
            if (!any_on_device && !ggml_cl_can_mul_mat(tensor->src[0], tensor->src[1], tensor)) {
                return false;
            }
            func = ggml_cl_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_mul_mat_id;
            break;
        case GGML_OP_SCALE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_scale;
            break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_nop;
            break;
        case GGML_OP_DIAG:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_diag;
            break;
        case GGML_OP_DIAG_MASK_INF:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_diag_mask_inf;
            break;
        case GGML_OP_SOFT_MAX:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_soft_max;
            break;
        case GGML_OP_ROPE:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_rope;
            break;
        case GGML_OP_SOLVE_TRI:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_solve_tri;
            break;
        case GGML_OP_IM2COL:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_im2col;
            break;
        case GGML_OP_ARGSORT:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_argsort;
            break;
        case GGML_OP_SUM_ROWS:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_sum_rows;
            break;
        case GGML_OP_CUMSUM:
            if (!any_on_device) {
                return false;
            }
            func = ggml_cl_cumsum;
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            if (!any_on_device) {
                return false;
            }
            ggml_cl_flash_attn(backend, tensor->src[0], tensor->src[1], tensor);
            return true;
        default:
            return false;
    }

    func(backend, tensor->src[0], tensor->src[1], tensor);
    return true;
}
