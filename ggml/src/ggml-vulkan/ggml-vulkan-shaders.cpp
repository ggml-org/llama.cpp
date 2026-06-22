#include "ggml-vulkan-common.h"

static bool ggml_vk_strip_decode_vector(const uint32_t * code, size_t word_count, std::vector<uint32_t> & out) {
    static const char kDecodeVectorExt[] = "SPV_NV_cooperative_matrix_decode_vector";

    if (word_count < 5) {
        return false;
    }

    bool uses_decode_vector = false;
    for (size_t pos = 5; pos < word_count; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::WordCountShift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_count);
        if (op == spv::OpExtension && wc >= 2) {
            const char * s = reinterpret_cast<const char *>(&code[pos + 1]);
            if (strcmp(s, kDecodeVectorExt) == 0) {
                uses_decode_vector = true;
                break;
            }
        }
        pos += wc;
    }

    if (!uses_decode_vector) {
        return false;
    }

    VK_LOG_DEBUG("ggml_vk_strip_decode_vector: stripping SPV_NV_cooperative_matrix_decode_vector");

    // Bulk-copy unchanged runs and only break the run when an instruction needs to
    // be dropped or patched. Use reserve + insert/push_back so the destination buffer
    // is touched exactly once (no zero-initialization pass from resize()).
    out.clear();
    out.reserve(word_count);

    size_t run_start = 0;
    auto flush_run = [&](size_t up_to) {
        if (up_to > run_start) {
            out.insert(out.end(), code + run_start, code + up_to);
        }
    };

    for (size_t pos = 5; pos < word_count; ) {
        uint32_t word = code[pos];
        uint32_t wc   = word >> spv::WordCountShift;
        uint32_t op   = word & spv::OpCodeMask;
        GGML_ASSERT(wc > 0 && pos + wc <= word_count);

        if (op == spv::OpExtension && wc >= 2) {
            const char * s = reinterpret_cast<const char *>(&code[pos + 1]);
            if (strcmp(s, kDecodeVectorExt) == 0) {
                flush_run(pos);
                pos += wc;
                run_start = pos;
                continue;
            }
        }

        if (op == spv::OpCapability && wc == 2 && code[pos + 1] == kSpvCapabilityCooperativeMatrixDecodeVectorNV) {
            flush_run(pos);
            pos += wc;
            run_start = pos;
            continue;
        }

        if (op == kSpvOpCooperativeMatrixLoadTensorNV) {
            // [opcode/wc][ResultType][Result][Pointer][Object][TensorLayout][MemOperand mask][mem extras...][TA mask][ta extras...]
            GGML_ASSERT(wc >= 8);

            uint32_t mem_mask = code[pos + 6];
            size_t   cur      = pos + 7;
            // Each of these MemoryAccess bits (when set) carries one trailing operand.
            cur += (mem_mask & 0x2)     ? 1 : 0; // Aligned
            cur += (mem_mask & 0x8)     ? 1 : 0; // MakePointerAvailable
            cur += (mem_mask & 0x10)    ? 1 : 0; // MakePointerVisible
            cur += (mem_mask & 0x10000) ? 1 : 0; // AliasScopeINTELMask
            cur += (mem_mask & 0x20000) ? 1 : 0; // NoAliasINTELMask
            GGML_ASSERT(cur < pos + wc);

            uint32_t ta_mask = code[cur];
            if ((ta_mask & kSpvTensorAddressingDecodeVectorFuncBit) == 0) {
                pos += wc;
                continue; // leave instruction inside the current unchanged run
            }

            flush_run(pos);

            // Append unchanged prefix of the instruction (header through the mem-extras).
            size_t inst_start = out.size();
            size_t pre_n      = cur - pos;
            out.insert(out.end(), code + pos, code + pos + pre_n);

            // Emit TA mask with the DecodeVectorFunc bit cleared.
            out.push_back(ta_mask & ~kSpvTensorAddressingDecodeVectorFuncBit);

            // TA extras: TensorView (0x1) and DecodeFunc (0x2) are kept verbatim;
            // DecodeVectorFunc (0x4) is dropped along with its trailing id operand.
            size_t keep_ta_extras = ((ta_mask & 0x1) ? 1 : 0) + ((ta_mask & 0x2) ? 1 : 0);
            if (keep_ta_extras) {
                out.insert(out.end(), code + cur + 1, code + cur + 1 + keep_ta_extras);
            }

            GGML_ASSERT(wc == pre_n + 1 + keep_ta_extras + 1);

            // Patch the instruction header with the new (one-shorter) word count.
            uint32_t new_wc = wc - 1;
            out[inst_start] = (new_wc << spv::WordCountShift) | op;

            pos += wc;
            run_start = pos;
            continue;
        }

        pos += wc;
    }

    flush_run(word_count);
    return true;
}

static void ggml_vk_create_pipeline_func(vk_device& device, vk_pipeline& pipeline, size_t spv_size, const void* spv_data, const std::string entrypoint,
                                         uint32_t parameter_count, std::array<uint32_t, 3> wg_denoms, std::vector<uint32_t> specialization_constants,
                                         bool disable_robustness, bool require_full_subgroups, uint32_t required_subgroup_size) {
    VK_LOG_DEBUG("ggml_vk_create_pipeline(" << device->name << ", " << pipeline->name << ", " << entrypoint << ", " << parameter_count <<
                 ", (" << wg_denoms[0] << "," << wg_denoms[1] << "," << wg_denoms[2] << "), specialization_constants, " <<
                 disable_robustness << ", " << require_full_subgroups << ", " << required_subgroup_size << ")");
    GGML_ASSERT(parameter_count > 0);
    GGML_ASSERT(parameter_count <= MAX_PARAMETER_COUNT);
    GGML_ASSERT(wg_denoms[0] > 0 && wg_denoms[1] > 0 && wg_denoms[2] > 0); // NOLINT

    vk::ShaderModuleCreateInfo shader_module_create_info({}, spv_size, reinterpret_cast<const uint32_t *>(spv_data));

    // Patch SPIR-V to enable RTE rounding for FP16, avoiding the need for
    // separate shader variants compiled with -DRTE16.
    std::vector<uint32_t> spirv;
    if (device->float_controls_rte_fp16) {
        const uint32_t* spv_words = reinterpret_cast<const uint32_t *>(spv_data);
        size_t word_count = spv_size / sizeof(uint32_t);
        spirv.assign(spv_words, spv_words + word_count);

        // Find insertion points respecting SPIR-V layout order:
        //   Header(5) -> OpCapability -> OpExtension -> ... -> OpEntryPoint -> OpExecutionMode -> ...
        size_t pos = 5; // skip header
        size_t cap_insert_pos = pos;
        size_t ext_insert_pos = pos;
        size_t exec_insert_pos = pos;
        uint32_t entry_point_id = 0;

        while (pos < spirv.size()) {
            uint32_t opcode = spirv[pos] & spv::OpCodeMask;
            uint32_t len    = spirv[pos] >> spv::WordCountShift;
            if (len == 0) break;

            if (opcode == spv::OpCapability) {
                cap_insert_pos = pos + len;
                ext_insert_pos = pos + len;
            } else if (opcode == spv::OpExtension) {
                ext_insert_pos = pos + len;
            } else if (opcode == spv::OpEntryPoint) {
                entry_point_id = spirv[pos + 2];
                exec_insert_pos = pos + len;
            } else if (opcode == spv::OpExecutionMode || opcode == spv::OpExecutionModeId) {
                exec_insert_pos = pos + len;
            } else if (entry_point_id != 0) {
                break;
            }

            pos += len;
        }

        // Insert from latest position first so earlier indices stay valid.

        // OpExecutionMode %entrypoint RoundingModeRTE 16
        uint32_t exec_mode[] = { (4u << spv::WordCountShift) | spv::OpExecutionMode, entry_point_id, spv::ExecutionModeRoundingModeRTE, 16 };
        spirv.insert(spirv.begin() + exec_insert_pos, std::begin(exec_mode), std::end(exec_mode));

        // OpExtension "SPV_KHR_float_controls"
        const char ext_str[] = "SPV_KHR_float_controls";
        size_t ext_str_words = CEIL_DIV(sizeof(ext_str), sizeof(uint32_t));
        std::vector<uint32_t> extension(1 + ext_str_words, 0);
        extension[0] = (uint32_t)((1 + ext_str_words) << spv::WordCountShift) | spv::OpExtension;
        memcpy(&extension[1], ext_str, sizeof(ext_str));
        spirv.insert(spirv.begin() + ext_insert_pos, extension.begin(), extension.end());

        // OpCapability RoundingModeRTE
        uint32_t capability[] = { (2u << spv::WordCountShift) | spv::OpCapability, spv::CapabilityRoundingModeRTE };
        spirv.insert(spirv.begin() + cap_insert_pos, std::begin(capability), std::end(capability));

        shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
    }

#if defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
    if (device->coopmat2 && !device->coopmat2_decode_vector) {
        const uint32_t * src   = spirv.empty() ? reinterpret_cast<const uint32_t *>(spv_data) : spirv.data();
        size_t           src_n = spirv.empty() ? spv_size / sizeof(uint32_t) : spirv.size();
        std::vector<uint32_t> stripped;
        if (ggml_vk_strip_decode_vector(src, src_n, stripped)) {
            spirv = std::move(stripped);
            shader_module_create_info = vk::ShaderModuleCreateInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
        }
    }
#endif

    pipeline->shader_module = device->device.createShaderModule(shader_module_create_info);

    vk::PushConstantRange pcr(
        vk::ShaderStageFlagBits::eCompute,
        0,
        pipeline->push_constant_size
    );

    vk::PipelineLayoutCreateInfo pipeline_layout_create_info(vk::PipelineLayoutCreateFlags(), device->dsl, pcr);
    pipeline->layout = device->device.createPipelineLayout(pipeline_layout_create_info);

    std::vector<vk::SpecializationMapEntry> specialization_entries(specialization_constants.size());

    for (size_t i = 0; i < specialization_constants.size(); i++) {
        specialization_entries[i].constantID = i;
        specialization_entries[i].offset = i * sizeof(uint32_t);
        specialization_entries[i].size = sizeof(uint32_t);
    }

    vk::SpecializationInfo specialization_info(
        specialization_entries.size(),
        specialization_entries.data(),
        specialization_constants.size() * sizeof(uint32_t),
        specialization_constants.data()
    );

    vk::PipelineShaderStageCreateFlags pipeline_shader_stage_create_flags{};

    if (device->subgroup_require_full_support && require_full_subgroups) {
        pipeline_shader_stage_create_flags |= vk::PipelineShaderStageCreateFlagBits::eRequireFullSubgroupsEXT;
    }

    vk::PipelineShaderStageCreateInfo pipeline_shader_create_info(
            pipeline_shader_stage_create_flags,
            vk::ShaderStageFlagBits::eCompute,
            pipeline->shader_module,
            entrypoint.c_str(),
            &specialization_info);

    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipeline_shader_stage_required_subgroup_size_create_info;
    pipeline_shader_stage_required_subgroup_size_create_info.requiredSubgroupSize = required_subgroup_size;
    if (device->subgroup_size_control && required_subgroup_size > 0) {
        GGML_ASSERT(device->subgroup_min_size <= required_subgroup_size && required_subgroup_size <= device->subgroup_max_size);
        pipeline_shader_create_info.setPNext(&pipeline_shader_stage_required_subgroup_size_create_info);
    }

    vk::ComputePipelineCreateInfo compute_pipeline_create_info(
        device->pipeline_executable_properties_support ?
            vk::PipelineCreateFlagBits::eCaptureStatisticsKHR :
            vk::PipelineCreateFlags{},
        pipeline_shader_create_info,
        pipeline->layout);

    vk::PipelineRobustnessCreateInfoEXT rci;

    if (device->pipeline_robustness && disable_robustness) {
        rci.storageBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        rci.uniformBuffers = vk::PipelineRobustnessBufferBehaviorEXT::eDisabled;
        compute_pipeline_create_info.setPNext(&rci);
    }

#if defined(VK_EXT_shader_64bit_indexing)
    vk::PipelineCreateFlags2CreateInfo pipelineFlags2CreateInfo;
    if (pipeline->is_64b_indexing)
    {
        pipelineFlags2CreateInfo.flags = vk::PipelineCreateFlagBits2::e64BitIndexingEXT;
        if (device->pipeline_executable_properties_support) {
            pipelineFlags2CreateInfo.flags |= vk::PipelineCreateFlagBits2::eCaptureStatisticsKHR;
        }
        pipelineFlags2CreateInfo.setPNext(compute_pipeline_create_info.pNext);
        compute_pipeline_create_info.setPNext(&pipelineFlags2CreateInfo);
    }
#endif

    try {
        pipeline->pipeline = device->device.createComputePipeline(VK_NULL_HANDLE, compute_pipeline_create_info).value;
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Compute pipeline creation failed for " << pipeline->name << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    if (vk_instance.debug_utils_support) {
        vk::DebugUtilsObjectNameInfoEXT duoni;
        duoni.objectType = vk::ObjectType::ePipeline;
        duoni.pObjectName = pipeline->name.c_str();
        duoni.objectHandle = /*reinterpret_cast*/(uint64_t)(static_cast<VkPipeline>(pipeline->pipeline));
        vk_instance.pfn_vkSetDebugUtilsObjectNameEXT(device->device, &static_cast<VkDebugUtilsObjectNameInfoEXT &>(duoni));
    }

    if (device->pipeline_executable_properties_support) {
        vk::PipelineExecutableInfoKHR executableInfo;
        executableInfo.pipeline = pipeline->pipeline;

        auto statistics = device->device.getPipelineExecutableStatisticsKHR(executableInfo);

        bool print_stats = !vk_pipeline_stats_filter.empty() &&
                           pipeline->name.find(vk_pipeline_stats_filter) != std::string::npos;
        if (print_stats) {
            std::cerr << "ggml_vulkan: pipeline stats for " << pipeline->name << ":" << std::endl;
        }

        for (auto & s : statistics) {
            if (print_stats) {
                std::cerr << "ggml_vulkan:   " << s.name.data() << ": ";
                switch (s.format) {
                    case vk::PipelineExecutableStatisticFormatKHR::eBool32:
                        std::cerr << (s.value.b32 ? "true" : "false");
                        break;
                    case vk::PipelineExecutableStatisticFormatKHR::eInt64:
                        std::cerr << s.value.i64;
                        break;
                    case vk::PipelineExecutableStatisticFormatKHR::eUint64:
                        std::cerr << s.value.u64;
                        break;
                    case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
                        std::cerr << s.value.f64;
                        break;
                }
                std::cerr << std::endl;
            }
            // "Register Count" is reported by NVIDIA drivers.
            if (strcmp(s.name, "Register Count") == 0) {
                VK_LOG_DEBUG(pipeline->name << " " << s.name << ": " << s.value.u64 << " registers");
                pipeline->register_count = (uint32_t)s.value.u64;
            }
        }
    }

    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        device->all_pipelines.push_back(pipeline);
        pipeline->compiled = true;
        pipeline->compile_pending = false;
    }
    device->compile_cv.notify_all();
}

void ggml_vk_destroy_pipeline(vk::Device& device, vk_pipeline& pipeline) {
    VK_LOG_DEBUG("ggml_pipeline_destroy_pipeline(" << pipeline->name << ")");
    device.destroyPipelineLayout(pipeline->layout);

    device.destroyShaderModule(pipeline->shader_module);

    device.destroyPipeline(pipeline->pipeline);
}

static vk_fa_tuning_params get_fa_tuning_params_scalar(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {

    vk_fa_tuning_params result{};
    result.path = FA_SCALAR;

    if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        // Disable subgroup use due to performance issues when enforcing subgroup sizes
        result.subgroup_size = 32;
        result.disable_subgroups = true;
    } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN) {
        result.subgroup_size = n_rows < 4 ? 32 : device->subgroup_size;
    } else {
        result.subgroup_size = device->subgroup_size;
    }

    // Row split splits the workgroup so that synchronization only has to happen within subgroups, which avoids barriers
    uint32_t row_split_max_hsk = 64;
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->architecture != AMD_GCN && !device->uma) {
        row_split_max_hsk = n_rows <= 8 ? 64 : 128;
    }
    result.row_split = (n_rows < 4 || hsk <= row_split_max_hsk) ? 1 : 4;

    if (result.subgroup_size > 32 && (n_rows < 4 || hsk < (result.row_split == 1 ? 128 : 64))) {
        result.workgroup_size = result.subgroup_size * 2;
    } else {
        result.workgroup_size = result.subgroup_size * 4;
    }

    const uint32_t D = hsk | hsv;

    const bool reduce_block_rows = D & 8 || n_kv < 1024 || device->vendor_id == VK_VENDOR_ID_INTEL;

    if (n_rows == 1) {
        result.block_rows = 1;
        result.block_cols = 64;
    } else {
        // row_split 1 means higher register use per row, so block size has to be adjusted
        if (result.row_split == 1) {
            result.block_rows = n_rows == 2 ? 2 : ((n_rows <= 4 || reduce_block_rows) ? 4 : 8);
        } else {
            result.block_rows = n_rows <= 4 ? 4 : ((n_rows <= 8 || reduce_block_rows) ? 8 : 16);
        }

        result.block_cols = (D & 8) ? 64 : 32;
    }

    const uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit

    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);

    result.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    if (!reduce_block_rows && !ggml_vk_flash_attn_scalar_shmem_support(device, result, hsk, hsv, f32acc, k_type, v_type)) {
        result.block_rows /= 2;
    }

    // On AMD RDNA, for small head sizes and big batch size the shader uses few registers, so too many subgroups get scheduled
    // at once and end up thrashing the cache. Fix this by setting a large (unused) shmem buffer that reduces occupancy.
    // This targets an occupancy of 4 subgroups per SIMD.
    if (device->vendor_id == VK_VENDOR_ID_AMD && device->properties.limits.maxComputeSharedMemorySize == 65536) {
        if (device->architecture != AMD_GCN && n_rows >= 64 && hsk <= 128) {
            // 30kb target for hsk > 64, 26kb for <= 64 due to smaller workgroup size
            // Values are guessed, tested on RDNA2
            result.limit_occupancy_shmem = (hsk <= 64 ? 26 : 30) * 1024 / 4 / 4;
        } else if (device->architecture == AMD_GCN && n_rows <= 8 && hsk >= 256) {
            // Same thing for GCN, with an occupancy target of 2 subgroups per SIMD.
            // Here low-batch FA with large head size is affected.
            // n_rows < 4 switch because workgroup size switches from 128 to 256 there.
            result.limit_occupancy_shmem = (n_rows < 4 ? 14 : 26) * 1024 / 4 / 4;
        }
    }

    return result;
}

static vk_fa_tuning_params get_fa_tuning_params_coopmat1(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    GGML_UNUSED(n_rows);
    GGML_UNUSED(n_kv);
    GGML_UNUSED(k_type);
    GGML_UNUSED(v_type);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params result{};
    result.path = FA_COOPMAT1;

    const uint32_t D = hsk | hsv;

    const uint32_t coopmat_block_rows = 16;
    const uint32_t coopmat_block_cols = 16;

    const uint32_t num_subgroups = 4;

    result.block_rows = coopmat_block_rows;
    result.block_cols = coopmat_block_cols * num_subgroups;
    result.row_split = num_subgroups;
    result.subgroup_size = device->subgroup_size;
    result.workgroup_size = num_subgroups * result.subgroup_size;

    const uint32_t D_lsb = D ^ (D & (D-1));  // extract lowest set bit
    result.d_split = std::min(std::min(result.subgroup_size, 8u), D_lsb / 4);

    result.shmem_staging = (device->vendor_id == VK_VENDOR_ID_NVIDIA && hsk < 256 && hsv < 256) ? 1 : 0;

    return result;
}

static vk_fa_tuning_params get_fa_tuning_params_coopmat2(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    GGML_UNUSED(n_kv);
    GGML_UNUSED(f32acc);

    vk_fa_tuning_params result{};
    result.path = FA_COOPMAT2;

    const uint32_t D = hsk | hsv;

    const bool small_rows = n_rows < 32;

    if (small_rows) {
        result.block_rows = 32;
        result.block_cols = 32;
    } else if (ggml_is_quantized(k_type) || ggml_is_quantized(v_type) || hsk >= 256 || hsv >= 256) {
        result.block_rows = (hsk >= 512 || hsv >= 512) ? 32 : 64;
        result.block_cols = 32;
    } else {
        result.block_rows = 64;
        result.block_cols = 64;
    }

    result.subgroup_size = device->subgroup_size;
    result.workgroup_size = (small_rows && (D % 32) == 0) ? 256 : 128;

    return result;
}

vk_fa_tuning_params get_fa_tuning_params(const vk_device& device, uint32_t hsk, uint32_t hsv, uint32_t n_rows, uint32_t n_kv, ggml_type k_type, ggml_type v_type, bool f32acc) {
    FaCodePath path = device->coopmat2 ? FA_COOPMAT2 :
                      device->coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR;

    if (path == FA_COOPMAT2 && k_type == GGML_TYPE_BF16 && !device->coopmat2_bf16_support) {
        path = FA_COOPMAT1;
    }
    if (path == FA_COOPMAT1 && k_type == GGML_TYPE_BF16 && !device->coopmat_bf16_support) {
        path = FA_SCALAR;
    }

    if (path == FA_COOPMAT1 && device->architecture == vk_device_architecture::NVIDIA_TURING) {
        // Nvidia compiler bug, see https://github.com/ggml-org/llama.cpp/pull/19075#issuecomment-3820716090
        path = FA_SCALAR;
    }

    if (path == FA_COOPMAT1) {
        bool shape_ok = (f32acc && device->coopmat_support_16x16x16_f32acc) ||
                        (!f32acc && device->coopmat_support_16x16x16_f16acc);
        const vk_fa_tuning_params params = get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
        bool shmem_ok = ggml_vk_flash_attn_coopmat_shmem_support(device, params, hsk, hsv, f32acc, k_type);

        if (!shape_ok || !shmem_ok) {
            path = FA_SCALAR;
        }
    }

    // scalar is faster than coopmat when N==1
    if (n_rows == 1 && (path == FA_COOPMAT1 || path == FA_COOPMAT2)) {
        path = FA_SCALAR;
    }

    // Q1_0 K/V is only implemented on coopmat2 (flash_attn_cm2); there is no scalar FA shader for it.
    if ((k_type == GGML_TYPE_Q1_0 || v_type == GGML_TYPE_Q1_0) && device->coopmat2) {
        path = FA_COOPMAT2;
    }

    switch (path) {
    case FA_SCALAR:
        return get_fa_tuning_params_scalar(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    case FA_COOPMAT1:
        return get_fa_tuning_params_coopmat1(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    case FA_COOPMAT2:
        return get_fa_tuning_params_coopmat2(device, hsk, hsv, n_rows, n_kv, k_type, v_type, f32acc);
    default:
        throw std::runtime_error("unsupported FaCodePath");
    }
}

vk_fa_pipeline_state get_fa_pipeline_state(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool aligned, bool f32acc,
                                                  bool use_mask, bool use_mask_opt, bool use_logit_softcap, ggml_type k_type, ggml_type v_type) {
    const bool old_amd_windows = device->vendor_id == VK_VENDOR_ID_AMD && device->driver_id == vk::DriverId::eAmdProprietary &&
                                 (device->architecture == AMD_GCN || device->architecture == AMD_RDNA1 || device->architecture == AMD_RDNA2);

    uint32_t flags = (use_mask_opt      ? 1 : 0) |
                     (use_mask          ? 2 : 0) |
                     (use_logit_softcap ? 4 : 0) |
                     (old_amd_windows   ? 8 : 0);

    const uint32_t subgroup_size = params.disable_subgroups ? 0 : params.subgroup_size;

    return vk_fa_pipeline_state{hsk, hsv, params.block_rows, params.block_cols, params.d_split, params.row_split, params.shmem_staging, params.path, params.workgroup_size, subgroup_size, aligned, f32acc, flags, params.limit_occupancy_shmem, k_type, v_type};
}

static std::vector<uint32_t> get_fa_spec_constants(const vk_fa_pipeline_state& state) {
    const auto fa_block_bytes = [](ggml_type t) -> uint32_t {
        if (t == GGML_TYPE_F32) return 16u;
        return (uint32_t) ggml_type_size(t);
    };
    return {
        /* 0 WorkGroupSize   */ state.workgroup_size,
        /* 1 Br              */ state.Br,
        /* 2 Bc              */ state.Bc,
        /* 3 HSK             */ state.HSK,
        /* 4 HSV             */ state.HSV,
        /* 5 Clamp           */ static_cast<uint32_t>(!state.aligned),
        /* 6 D_split         */ state.D_split,
        /* 7 row_split       */ state.row_split,
        /* 8 SubGroupSize    */ state.subgroup_size,
        /* 9 SHMEM_STAGING   */ state.shmem_staging ? 1u : 0u,
        /*10 Flags           */ state.flags,
        /*11 LIMIT_OCCUPANCY_SHMEM */ state.limit_occupancy_shmem,
        /*12 FaTypeK         */ static_cast<uint32_t>(state.k_type),
        /*13 FaTypeV         */ static_cast<uint32_t>(state.v_type),
        /*14 FaBlockBytesK   */ fa_block_bytes(state.k_type),
        /*15 FaBlockBytesV   */ fa_block_bytes(state.v_type),
    };
}

static bool ggml_vk_matmul_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    uint32_t lut_size = 0;
    switch (src0_type) {
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
        // Regular matmul uses the compact uint16_t IQ1 grid; the expanded
        // uint32_t grid is only enabled for the q8_1/int-dot vector path.
        lut_size = 2*2048;
        break;
    case GGML_TYPE_IQ2_XXS:
        lut_size = 8*256;
        break;
    case GGML_TYPE_IQ2_XS:
        lut_size = 8*512;
        break;
    case GGML_TYPE_IQ2_S:
        lut_size = 8*1024;
        break;
    case GGML_TYPE_IQ3_XXS:
        lut_size = 4*256;
        break;
    case GGML_TYPE_IQ3_S:
        lut_size = 4*512;
        break;
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_MXFP4:
        lut_size = 4*16;
        break;
    case GGML_TYPE_NVFP4:
        // Same kvalues budget as MXFP4 plus ue4m3_fp32_lut[128] (types.glsl, DATA_A_NVFP4).
        lut_size = 4*16 + 128u * (uint32_t)sizeof(float);
        break;
    default:
        break;
    }

    // Needs to be kept up to date on shader changes
    const uint32_t bank_conflict_offset = device->coopmat_support ? 8 : 1;
    const uint32_t type_size = device->fp16 ? sizeof(ggml_fp16_t) : sizeof(float);
    const uint32_t warps = warptile[0] / warptile[10];

    const uint32_t load_bufs = (warptile[1] + warptile[2]) * (warptile[3] + bank_conflict_offset) * type_size;
    const uint32_t mmid_row_ids = mul_mat_id ? (warptile[2] * 2 * sizeof(uint16_t)) : 0;
    const uint32_t coopmat_stage = device->coopmat_support ? warptile[7] * warptile[8] / warps * sizeof(float) : 0;
    const uint32_t ballots_sh = mul_mat_id ? (warps * 4 * sizeof(uint32_t)) : 0;

    const uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size + ballots_sh;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", supported=" << supported);

    return supported;
}

static bool ggml_vk_matmul_int_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type) {

    // FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float.
    const uint32_t fp_size   = device->fp16 ? 2u : 4u;
    const uint32_t fp_align  = fp_size;
    const uint32_t fp2_size  = 2u * fp_size;
    const uint32_t fp2_align = device->fp16 ? 4u : 8u;

    struct member { uint32_t size, align; };
    auto std430_size = [](std::initializer_list<member> members) {
        uint32_t off = 0, struct_align = 1;
        for (const auto &m : members) {
            off = (off + m.align - 1) & ~(m.align - 1);
            off += m.size;
            struct_align = std::max(struct_align, m.align);
        }
        return (off + struct_align - 1) & ~(struct_align - 1);
    };

    uint32_t block_a_size = 0;
    switch (src0_type) {
        case GGML_TYPE_Q4_0:    block_a_size = std430_size({{16, 4}, {fp_size,  fp_align}});                  break; // qs[16/4] + dm
        case GGML_TYPE_Q4_1:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[16/4] + dm(vec2)
        case GGML_TYPE_Q5_0:    block_a_size = std430_size({{16, 4}, {4, 4}, {fp_size,  fp_align}});          break; // qs[16/4] + qh + dm
        case GGML_TYPE_Q5_1:    block_a_size = std430_size({{16, 4}, {4, 4}, {fp2_size, fp2_align}});         break; // qs[16/4] + qh + dm(vec2)
        case GGML_TYPE_Q8_0:    block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + dm
        case GGML_TYPE_MXFP4:   block_a_size = std430_size({{32, 4}, {fp_size,  fp_align}});                  break; // qs[8] + d
        case GGML_TYPE_Q2_K:    block_a_size = std430_size({{ 8, 4}, {2, 2}, {fp2_size, fp2_align}});         break; // qs[2] + scales(u8vec2) + dm(vec2)
        case GGML_TYPE_Q3_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[4] + d_scales(vec2)
        case GGML_TYPE_Q4_K:    block_a_size = std430_size({{16, 4}, {fp2_size, fp2_align}});                 break; // qs[4] + dm(vec2)
        case GGML_TYPE_Q5_K:    block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 break; // qs[8] + dm(vec2)
        case GGML_TYPE_Q6_K:    block_a_size = std430_size({{32, 4}, {fp2_size, fp2_align}});                 break; // qs[8] + d_scales(vec2)
        default:
            return false;
    }

    // block_b_cache: { int32_t qs[8]; FLOAT_TYPEV2 ds; }
    const uint32_t block_b_size = std430_size({{32, 4}, {fp2_size, fp2_align}});

    const uint32_t BM = warptile[1];
    const uint32_t BN = warptile[2];
    // mul_mmq.comp: BK_STEP=1 for MUL_MAT_ID, 4 otherwise.
    const uint32_t BK_STEP = mul_mat_id ? 1u : 4u;

    const uint32_t buf_a_size = BM * BK_STEP * block_a_size;
    const uint32_t buf_b_size = BN * BK_STEP * block_b_size;
    const uint32_t mmid_row_ids = mul_mat_id ? (BN * 2u * (uint32_t)sizeof(uint16_t)) : 0u;

    const uint32_t warps = warptile[0] / warptile[10];
    const uint32_t ballots_sh = mul_mat_id ? (warps * 4u * (uint32_t)sizeof(uint32_t)) : 0u;

    const uint32_t total_size = buf_a_size + buf_b_size + mmid_row_ids + ballots_sh;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_matmul_int_shmem_support(warptile=(" << warptile[0] << "," << warptile[1] << "," << warptile[2] << "), "
                 "mul_mat_id=" << mul_mat_id << ", src0_type=" << ggml_type_name(src0_type) << ", total=" << total_size << ", supported=" << supported);

    return supported;
}

static const std::unordered_map<std::string, uint32_t> rdna1_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
    {"argmax", 64}, {"mul_mat_vec", 64},
    {"mul_mat_vec_f16", 32}, {"mul_mat_vec_f32_f16", 32}
};

static const std::unordered_map<std::string, uint32_t> rdna2_pipelines = {
    {"soft_max", 64}, {"im2col", 64},
};

static std::vector<GpuPipelineConfig> gpu_pipeline_configs = {
    {
        vk_device_architecture::AMD_RDNA1,
        {
            rdna1_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
    {
        vk_device_architecture::AMD_RDNA2,
        {
            rdna2_pipelines,
        },
        RDNA_DEFAULT_SUBGROUP_SIZE
    },
};

uint32_t get_subgroup_size(const std::string &pipeline_name, const vk_device_architecture &arch) {
    for (const auto &config : gpu_pipeline_configs) {
        if (config.arch == arch) {
            auto pipIt = config.pipelines.find(pipeline_name);
            if (pipIt != config.pipelines.end()) {
                return pipIt->second;
            }
            std::vector<std::pair<std::string, uint32_t>> sorted_pipelines(config.pipelines.begin(), config.pipelines.end());
            std::sort(sorted_pipelines.begin(), sorted_pipelines.end(),
                      [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
            for (const auto &entry : sorted_pipelines) {
                if (pipeline_name.find(entry.first) != std::string::npos) {
                    return entry.second;
                }
            }
            return config.default_subgroup_size;
        }
    }
    return 0; // If no matching configuration is found
}

static bool ggml_vk_fa_scalar_uses_mmq(const vk_device& device, ggml_type k_type) {
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    return device->integer_dot_product && device->subgroup_clustered &&
           (k_type == GGML_TYPE_Q4_0 || k_type == GGML_TYPE_Q4_1 ||
            k_type == GGML_TYPE_Q5_0 || k_type == GGML_TYPE_Q5_1 ||
            k_type == GGML_TYPE_Q8_0);
#else
    GGML_UNUSED(device);
    GGML_UNUSED(k_type);
    return false;
#endif
}

void ggml_vk_load_shaders(vk_device& device, vk_pipeline requested) {
    VK_LOG_DEBUG("ggml_vk_load_shaders(" << device->name << ")");

    // some shaders have a minimum subgroup size
    const uint32_t subgroup_size_8 = std::max(device->subgroup_size, 8u);
    const uint32_t subgroup_size_16 = std::max(device->subgroup_size, 16u);
    const uint32_t subgroup_size_32 = std::max(device->subgroup_size, 32u);

    const uint32_t mul_mat_subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
    const uint32_t mul_mat_subgroup_size_8 = std::max(mul_mat_subgroup_size, 8u);
    const uint32_t mul_mat_subgroup_size_16 = std::max(mul_mat_subgroup_size, 16u);
    const uint32_t mul_mat_subgroup_size_32 = std::max(mul_mat_subgroup_size, 32u);

    const bool subgroup_min_size_16 = (!device->subgroup_size_control && device->subgroup_size >= 16) ||
                                      (device->subgroup_size_control && device->subgroup_max_size >= 16);

    // mulmat
    std::vector<uint32_t> l_warptile, m_warptile, s_warptile,
                          l_warptile_id, m_warptile_id, s_warptile_id,
                          l_warptile_mmq, m_warptile_mmq, s_warptile_mmq,
                          l_warptile_mmq_int, m_warptile_mmq_int, s_warptile_mmq_int,
                          l_warptile_mmq_int_k, m_warptile_mmq_int_k, s_warptile_mmq_int_k,
                          l_warptile_mmq_k, m_warptile_mmq_k, s_warptile_mmq_k,
                          l_warptile_mmqid, m_warptile_mmqid, s_warptile_mmqid,
                          l_warptile_mmqid_int, m_warptile_mmqid_int, s_warptile_mmqid_int,
                          l_warptile_mmqid_int_k, m_warptile_mmqid_int_k, s_warptile_mmqid_int_k;
    std::array<uint32_t, 3> l_wg_denoms, m_wg_denoms, s_wg_denoms,
                            l_mmq_wg_denoms, m_mmq_wg_denoms, s_mmq_wg_denoms,
                            l_mmq_wg_denoms_k, m_mmq_wg_denoms_k, s_mmq_wg_denoms_k,
                            l_mmqid_wg_denoms, m_mmqid_wg_denoms, s_mmqid_wg_denoms;

    uint32_t l_align, m_align, s_align;

    vk_pipeline wait_pipeline;
    CompileTask claimed_task {};
    bool has_claimed_task = false;

    // The rest of the walk reads and writes shared device state, so hold the
    // lock until we're done deciding what to compile.
    std::unique_lock<std::mutex> compile_lock(device->compile_mutex);

    if (device->coopmat2) {
        // spec constants and tile sizes for non-quant matmul/matmul_id
        l_warptile = { 256, 128, 256, 64, 1 };
        m_warptile = { 256, 128, 128, 64, 0 };
        s_warptile = { 128,  64,  64, 64, 0 };
        l_wg_denoms = {128, 256, 1 };
        m_wg_denoms = {128, 128, 1 };
        s_wg_denoms = { 64,  64, 1 };

        // spec constants and tile sizes for quant matmul (non-Qi_K)
        l_warptile_mmq = { 256, 128, 256, 64, 1 };
        m_warptile_mmq = { 256, 128, 128, 64, 1 };
        s_warptile_mmq = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms = { 128, 256, 1 };
        m_mmq_wg_denoms = { 128, 128, 1 };
        s_mmq_wg_denoms = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul (Qi_K)
        l_warptile_mmq_k = { 256, 128, 256, 64, 1 };
        m_warptile_mmq_k = { 256, 128, 128, 64, 1 };
        s_warptile_mmq_k = { 256, 32,  64, 128, 0 };
        l_mmq_wg_denoms_k = { 128, 256, 1 };
        m_mmq_wg_denoms_k = { 128, 128, 1 };
        s_mmq_wg_denoms_k = { 32,  64,  1 };

        // spec constants and tile sizes for quant matmul_id
        const uint32_t mmqid_bk = device->coopmat2_decode_vector ? 64u : 32u;
        l_warptile_mmqid = { 256, 128, 128, mmqid_bk, 1, device->subgroup_size };
        m_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0, device->subgroup_size };
        s_warptile_mmqid = { 256, 128, 64,  mmqid_bk, 0, device->subgroup_size };
        l_mmqid_wg_denoms = { 128, 128, 1 };
        m_mmqid_wg_denoms = { 128, 64, 1 };
        s_mmqid_wg_denoms = { 128, 64, 1 };

        l_align = 128;
        m_align =  64;
        s_align =  32;
    } else {
        // Matrix cores require different warp group sizes
        const uint32_t tm_l = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_m = device->coopmat_support ? device->coopmat_m : 4;
        const uint32_t tm_s = device->coopmat_support ? device->coopmat_m : 2;
        const uint32_t tn_l = device->coopmat_support ? device->coopmat_n : 4;
        const uint32_t tn_m = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tn_s = device->coopmat_support ? device->coopmat_n : 2;
        const uint32_t tk_l = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_m = device->coopmat_support ? device->coopmat_k : 1;
        const uint32_t tk_s = device->coopmat_support ? device->coopmat_k : 1;

        const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;

        l_warptile = { 128,             128, 128, 16, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
        m_warptile = { 128,              64,  64, 16, subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        s_warptile = { subgroup_size_32, 32,  32, 16, s_warptile_wm,       32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        l_warptile_mmq = { 128,             128, 128, 32, subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, subgroup_size_8 };
        m_warptile_mmq = { 128,              64,  64, 32, subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        s_warptile_mmq = { subgroup_size_32, 32,  32, 32, s_warptile_wm,       32, 2, tm_s, tn_s, tk_s, subgroup_size_8 };

        // Integer MMQ has a smaller shared memory profile, but heavier register use
        l_warptile_mmq_int = { 128,             128, 128, 32, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        m_warptile_mmq_int = { 128,              64,  64, 32, subgroup_size_8,     32, 2, 2, 2, 1, subgroup_size_8 };
        s_warptile_mmq_int = { subgroup_size_32, 32,  32, 32, s_warptile_wm,       32, 2, 2, 1, 1, subgroup_size_8 };

        // K-quants use even more registers, mitigate by setting WMITER to 1
        l_warptile_mmq_int_k = { 128,               128, 128, 32, subgroup_size_8 * 2, 64, 1, 4, 4, 1, subgroup_size_8 };
        m_warptile_mmq_int_k = { 128,                64,  64, 32, subgroup_size_8,     32, 1, 2, 2, 1, subgroup_size_8 };
        s_warptile_mmq_int_k = { subgroup_size_32,   32,  32, 32, s_warptile_wm,       32, 1, 2, 1, 1, subgroup_size_8 };

        l_warptile_id = { 128,                      128, 128, 16, mul_mat_subgroup_size_16 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_16 };
        m_warptile_id = { 128,                       64,  64, 16, mul_mat_subgroup_size_16,     32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_16 };
        s_warptile_id = { mul_mat_subgroup_size_16,  32,  32, 16, s_warptile_wm,                32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_16 };

        l_warptile_mmqid = { 128,                       128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, tm_l, tn_l, tk_l, mul_mat_subgroup_size_8 };
        m_warptile_mmqid = { 128,                        64,  64, 32, mul_mat_subgroup_size_8,     32, 2, tm_m, tn_m, tk_m, mul_mat_subgroup_size_8 };
        s_warptile_mmqid = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,               32, 2, tm_s, tn_s, tk_s, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int = { 128,                       128, 128, 32, mul_mat_subgroup_size_8 * 2, 64, 2, 4, 4, 1, mul_mat_subgroup_size_8 };
        m_warptile_mmqid_int = { 128,                        64,  64, 32, mul_mat_subgroup_size_8,     32, 2, 2, 2, 1, mul_mat_subgroup_size_8 };
        s_warptile_mmqid_int = { mul_mat_subgroup_size_32,   32,  32, 32, s_warptile_wm,               32, 2, 2, 1, 1, mul_mat_subgroup_size_8 };

        l_warptile_mmqid_int_k = { 128,                     128, 128, 32, mul_mat_subgroup_size_16 * 2, 64, 1, 4, 4, 1, mul_mat_subgroup_size_16 };
        m_warptile_mmqid_int_k = { 128,                      64,  64, 32, mul_mat_subgroup_size_16,     32, 1, 2, 2, 1, mul_mat_subgroup_size_16 };
        s_warptile_mmqid_int_k = { mul_mat_subgroup_size_32, 32,  32, 32, s_warptile_wm,                32, 1, 2, 1, 1, mul_mat_subgroup_size_16 };

        // chip specific tuning
        if ((device->architecture == AMD_GCN) && (device->driver_id != vk::DriverId::eAmdProprietary)) {
            m_warptile_mmq = m_warptile_mmq_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
            m_warptile_mmqid = m_warptile_mmqid_int = { 256, 64, 64, 32, 16, 16, 2, 2, 2, 1, 16 };
        } else if (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != vk::DriverId::eAmdProprietary) {
            // This is intentionally using tx_m values, slight performance increase
            l_warptile = { 256, 128, 128, 16, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
            l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, subgroup_size_8, 64, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
            l_warptile_mmq_int_k = { 256, 128, 128, 32, subgroup_size_16, 64, 1, 4, 2, 1, subgroup_size_16 };
        } else if (device->vendor_id == VK_VENDOR_ID_INTEL && device->coopmat_support && device->architecture == INTEL_XE2) {
            // Xe2/Xe3 with coopmat enabled - warptile performance tuning
            l_warptile = { 512, 128, 128, 16, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
            l_warptile_mmq = { 512, 128, 128, 32, subgroup_size_8, 32, 2, tm_m, tn_m, tk_m, subgroup_size_8 };
        }

        l_mmq_wg_denoms = l_wg_denoms = {128, 128, 1 };
        m_mmq_wg_denoms = m_wg_denoms = { 64,  64, 1 };
        s_mmq_wg_denoms = s_wg_denoms = { 32,  32, 1 };
        l_align = 128;
        m_align =  64;
        s_align =  32;

        for (uint32_t i = 0; i < GGML_TYPE_COUNT; ++i) {
            ggml_type t = (ggml_type)i;
            // Disable medium and large matrix multiplication if not enough shared memory is available
            // Check mmq warptiles as the largest configuration
            // Throw an error if not enough for any matrix multiplication is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmq, false, t)) {
                std::cerr << "ggml_vulkan: Error: Shared memory size too small for matrix multiplication." << std::endl;
                throw std::runtime_error("Shared memory size too small for matrix multiplication.");
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmq, false, t)) {
                device->mul_mat_m[i] = false;
                device->mul_mat_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmq, false, t)) {
                device->mul_mat_l[i] = false;
            }

            // Disable mul_mat_id if not enough shared memory is available
            if (!ggml_vk_matmul_shmem_support(device, s_warptile_mmqid, true, t)) {
                device->mul_mat_id_s[i] = false;
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, m_warptile_mmqid, true, t)) {
                device->mul_mat_id_m[i] = false;
                device->mul_mat_id_l[i] = false;
            } else if (!ggml_vk_matmul_shmem_support(device, l_warptile_mmqid, true, t)) {
                device->mul_mat_id_l[i] = false;
            }

            // The q8_1 mmq path has its own (larger) shmem layout, check it separately.
            // K-quants use the _int_k warptiles, others use _int.
            const bool is_k_quant = (t == GGML_TYPE_Q2_K || t == GGML_TYPE_Q3_K ||
                                     t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K ||
                                     t == GGML_TYPE_Q6_K);
            const auto & s_int   = is_k_quant ? s_warptile_mmq_int_k   : s_warptile_mmq_int;
            const auto & m_int   = is_k_quant ? m_warptile_mmq_int_k   : m_warptile_mmq_int;
            const auto & l_int   = is_k_quant ? l_warptile_mmq_int_k   : l_warptile_mmq_int;
            const auto & s_intid = is_k_quant ? s_warptile_mmqid_int_k : s_warptile_mmqid_int;
            const auto & m_intid = is_k_quant ? m_warptile_mmqid_int_k : m_warptile_mmqid_int;
            const auto & l_intid = is_k_quant ? l_warptile_mmqid_int_k : l_warptile_mmqid_int;

            if (!ggml_vk_matmul_int_shmem_support(device, s_int, false, t)) {
                device->mul_mat_s_int[i] = false;
                device->mul_mat_m_int[i] = false;
                device->mul_mat_l_int[i] = false;
            } else if (!ggml_vk_matmul_int_shmem_support(device, m_int, false, t)) {
                device->mul_mat_m_int[i] = false;
                device->mul_mat_l_int[i] = false;
            } else if (!ggml_vk_matmul_int_shmem_support(device, l_int, false, t)) {
                device->mul_mat_l_int[i] = false;
            }

            if (!ggml_vk_matmul_int_shmem_support(device, s_intid, true, t)) {
                device->mul_mat_id_s_int[i] = false;
                device->mul_mat_id_m_int[i] = false;
                device->mul_mat_id_l_int[i] = false;
            } else if (!ggml_vk_matmul_int_shmem_support(device, m_intid, true, t)) {
                device->mul_mat_id_m_int[i] = false;
                device->mul_mat_id_l_int[i] = false;
            } else if (!ggml_vk_matmul_int_shmem_support(device, l_intid, true, t)) {
                device->mul_mat_id_l_int[i] = false;
            }
        }
    }

    if (!device->pipeline_matmul_f32) {
        device->pipeline_matmul_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_f32_f16) {
        device->pipeline_matmul_f32_f16 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_id_f32) {
        device->pipeline_matmul_id_f32 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_bf16) {
        device->pipeline_matmul_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
    }
    if (!device->pipeline_matmul_id_bf16) {
        device->pipeline_matmul_id_bf16 = std::make_shared<vk_matmul_pipeline_struct>();
    }

    auto const &ggml_vk_create_pipeline = [&](vk_device& device, vk_pipeline& base_pipeline, const char *name, size_t spv_size, const void* spv_data, const char *entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {

        if (!require_full_subgroups && required_subgroup_size == 0) {
            required_subgroup_size = get_subgroup_size(name, device->architecture);
        }

        vk_pipeline *ptr = &base_pipeline;

        int num_pipelines = 1;
#if defined(VK_EXT_shader_64bit_indexing)
        if (device->shader_64b_indexing) {
            num_pipelines = 2;
        }
#endif
        for (int i = 0; i < num_pipelines; ++i, ptr = &(*ptr)->next) {
            vk_pipeline &pipeline = *ptr;
            if (!pipeline) {
                pipeline = std::make_shared<vk_pipeline_struct>();
            }
            if (!pipeline->initialized) {
                pipeline->name = name;
                pipeline->parameter_count = parameter_count;
                pipeline->push_constant_size = push_constant_size;
                pipeline->wg_denoms = wg_denoms;
                pipeline->align = align;
                pipeline->initialized = true;
#if defined(VK_EXT_shader_64bit_indexing)
                pipeline->is_64b_indexing = (i == 1);
#endif
            }

            // We only care about the pipeline this call asked for; the rest
            // (including the 64-bit indexing variant) are handled by their
            // own request_descriptor_sets / load_shaders calls.
            if (pipeline.get() != requested.get()) {
                continue;
            }

            if (pipeline->compiled) {
                continue;
            }

            wait_pipeline = pipeline;

            if (!pipeline->compile_pending) {
                pipeline->compile_pending = true;
                claimed_task.pipeline = pipeline;
                claimed_task.spv_size = spv_size;
                claimed_task.spv_data = spv_data;
                claimed_task.entrypoint = entrypoint;
                claimed_task.parameter_count = parameter_count;
                claimed_task.wg_denoms = wg_denoms;
                claimed_task.specialization_constants = specialization_constants;
                claimed_task.disable_robustness = disable_robustness;
                claimed_task.require_full_subgroups = require_full_subgroups;
                claimed_task.required_subgroup_size = required_subgroup_size;
                has_claimed_task = true;
            }
        }
    };

    auto const &ggml_vk_create_pipeline2 = [&](vk_device& device, vk_pipeline& pipeline, const std::string &name, size_t spv_size, const void* spv_data, const char *entrypoint,
                                              uint32_t parameter_count, uint32_t push_constant_size, std::array<uint32_t, 3> wg_denoms, const std::vector<uint32_t>& specialization_constants,
                                              uint32_t align, bool disable_robustness = false, bool require_full_subgroups = false, uint32_t required_subgroup_size = 0) {
        return ggml_vk_create_pipeline(device, pipeline, name.c_str(), spv_size, spv_data, entrypoint,
                                       parameter_count, push_constant_size, wg_denoms, specialization_constants,
                                       align, disable_robustness, require_full_subgroups, required_subgroup_size);
    };

    // FA scalar has two SPIR-V modules (MMQ vs non-MMQ); FA cm1 has one. K/V
    // quant type is selected at runtime via the FaTypeK / FaTypeV spec constants.

    for (auto &fa : device->pipeline_flash_attn_f32_f16) {
        if (fa.first.path != FA_SCALAR) continue;
        const uint32_t Br = fa.first.Br;
        const uint32_t Bc = fa.first.Bc;
        const bool aligned = fa.first.aligned;
        const bool f32acc = fa.first.f32acc;
        const uint32_t fa_sgs = fa.first.subgroup_size;
        const bool fa_ds = fa.first.subgroup_size == 0;

        const bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
        const bool use_mmq = ggml_vk_fa_scalar_uses_mmq(device, fa.first.k_type);
        const void * spv_data = nullptr;
        size_t spv_size = 0;
        const char *name = nullptr;
        if (bf16_kv) {
            spv_data = flash_attn_f32_f16_fp32_data;
            spv_size = flash_attn_f32_f16_fp32_len;
            name = aligned ? "flash_attn_f32_bf16_aligned" : "flash_attn_f32_bf16";
        } else if (use_mmq) {
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->fp16) {
                if (f32acc) { spv_data = flash_attn_f32_f16_int8_data;        spv_size = flash_attn_f32_f16_int8_len; }
                else        { spv_data = flash_attn_f32_f16_f16acc_int8_data; spv_size = flash_attn_f32_f16_f16acc_int8_len; }
            } else {
                spv_data = flash_attn_f32_f16_fp32_int8_data;
                spv_size = flash_attn_f32_f16_fp32_int8_len;
            }
#endif
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        } else {
            if (device->fp16) {
                if (device->dot2_f16) {
                    if (f32acc) { spv_data = flash_attn_f32_f16_dot2_data;        spv_size = flash_attn_f32_f16_dot2_len; }
                    else        { spv_data = flash_attn_f32_f16_dot2_f16acc_data; spv_size = flash_attn_f32_f16_dot2_f16acc_len; }
                } else {
                    if (f32acc) { spv_data = flash_attn_f32_f16_data;        spv_size = flash_attn_f32_f16_len; }
                    else        { spv_data = flash_attn_f32_f16_f16acc_data; spv_size = flash_attn_f32_f16_f16acc_len; }
                }
            } else {
                spv_data = flash_attn_f32_f16_fp32_data;
                spv_size = flash_attn_f32_f16_fp32_len;
            }
            name = aligned ? "flash_attn_f32_f16_aligned" : "flash_attn_f32_f16";
        }
        ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 7,
                                sizeof(vk_flash_attn_push_constants), {Br, 1, 1},
                                get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                !fa_ds, !fa_ds ? fa_sgs : 0);
    }

#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat1_fa_support) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16) {
            if (fa.first.path != FA_COOPMAT1) continue;
            const uint32_t Br = fa.first.Br;
            const uint32_t Bc = fa.first.Bc;
            const bool aligned = fa.first.aligned;
            const bool f32acc = fa.first.f32acc;
            const uint32_t fa_sgs = fa.first.subgroup_size;
            const bool fa_ds = fa.first.subgroup_size == 0;

            const bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;

            const void * spv_data;
            size_t spv_size;
            const char *name;
            if (bf16_kv) {
#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                if (!device->coopmat_bf16_support) continue;
                spv_data = flash_attn_f32_f16_bf16_cm1_data;
                spv_size = flash_attn_f32_f16_bf16_cm1_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm1" : "flash_attn_f32_bf16_cm1";
#else
                continue;
#endif
            } else {
                if (f32acc) { spv_data = flash_attn_f32_f16_cm1_data;        spv_size = flash_attn_f32_f16_cm1_len; }
                else        { spv_data = flash_attn_f32_f16_f16acc_cm1_data; spv_size = flash_attn_f32_f16_f16acc_cm1_len; }
                name = aligned ? "flash_attn_f32_f16_aligned_cm1" : "flash_attn_f32_f16_cm1";
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 7,
                                    sizeof(vk_flash_attn_push_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, true,
                                    !fa_ds, !fa_ds ? fa_sgs : 0);
        }
    }
#endif

#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {
        for (auto &fa : device->pipeline_flash_attn_f32_f16) {
            if (fa.first.path != FA_COOPMAT2) continue;
            const uint32_t Br = fa.first.Br;
            const uint32_t Bc = fa.first.Bc;
            const bool aligned = fa.first.aligned;
            const bool f32acc = fa.first.f32acc;

            const bool bf16_kv = fa.first.k_type == GGML_TYPE_BF16;
            const void * spv_data;
            size_t spv_size;
            const char * name;
            if (bf16_kv) {
#if defined(VK_KHR_shader_bfloat16) && defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
                if (!device->coopmat2_bf16_support) continue;
                spv_data = flash_attn_f32_f16_bf16_cm2_data;
                spv_size = flash_attn_f32_f16_bf16_cm2_len;
                name = aligned ? "flash_attn_f32_bf16_aligned_cm2" : "flash_attn_f32_bf16_cm2";
#else
                continue;
#endif
            } else if (aligned) {
                if (f32acc) { spv_data = flash_attn_f32_f16_cm2_data;        spv_size = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_aligned_f32acc_cm2"; }
                else        { spv_data = flash_attn_f32_f16_f16acc_cm2_data; spv_size = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_aligned_f16acc_cm2"; }
            } else {
                if (f32acc) { spv_data = flash_attn_f32_f16_cm2_data;        spv_size = flash_attn_f32_f16_cm2_len;        name = "flash_attn_f32_f16_f32acc_cm2"; }
                else        { spv_data = flash_attn_f32_f16_f16acc_cm2_data; spv_size = flash_attn_f32_f16_f16acc_cm2_len; name = "flash_attn_f32_f16_f16acc_cm2"; }
            }
            ggml_vk_create_pipeline(device, fa.second, name, spv_size, spv_data, "main", 7,
                                    sizeof(vk_flash_attn_push_constants), {Br, 1, 1},
                                    get_fa_spec_constants(fa.first), aligned ? Bc : 1, true, false, 0);
        }
    }
#endif

    const int mul_mat_id_param_count = 5;

#if defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    if (device->coopmat2) {

        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm2_len, NAMELC ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, true);   \
        ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm2_len, NAMELC ## _aligned ## F16ACC ## _cm2_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, true);   \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT) \
        CREATE_MM(PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \
        CREATE_MM(PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT)   \

        CREATE_MM2(pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3)
        }
#endif
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q1_0], matmul_q1_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_0], matmul_q4_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_1], matmul_q4_1_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_0], matmul_q5_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_1], matmul_q5_1_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q8_0], matmul_q8_0_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q2_K], matmul_q2_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q3_K], matmul_q3_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q4_K], matmul_q4_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q5_K], matmul_q5_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_Q6_K], matmul_q6_k_f16, mmq_wg_denoms_k, warptile_mmq_k, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ1_S],   matmul_iq1_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ1_M],   matmul_iq1_m_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ2_S],   matmul_iq2_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f16, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ3_S],   matmul_iq3_s_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f16,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_MXFP4],   matmul_mxfp4_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_f16[GGML_TYPE_NVFP4],   matmul_nvfp4_f16,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3)

        GGML_ASSERT(device->subgroup_ballot);

        CREATE_MM2(pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, 5)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, 5)
        }
#endif
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_subgroup_q1_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_subgroup_iq1_s_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_subgroup_iq1_m_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_subgroup_iq2_xxs_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_subgroup_iq2_xs_f16,  mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_subgroup_iq2_s_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_subgroup_iq3_xxs_f16, mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_subgroup_iq3_s_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_subgroup_iq4_xs_f16,  mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_subgroup_iq4_nl_f16,  mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_subgroup_mxfp4_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
        CREATE_MM2(pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_subgroup_nvfp4_f16,   mmqid_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, 5)
#undef CREATE_MM
#undef CREATE_MM2
    } else
#endif  // defined(VK_NV_cooperative_matrix2) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->coopmat_support) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _cm1_len, NAMELC ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, true);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, true);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, true);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _cm1_len, NAMELC ## _aligned ## F16ACC ## _cm1_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, true);   \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->coopmat_acc_f16_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \
        if (device->coopmat_acc_f32_support) { \
            CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        } \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16_f32, matmul_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 3, );
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, )
        }
#endif

        if (device->coopmat_acc_f16_support) {
            CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0], matmul_q1_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0], matmul_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1], matmul_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0], matmul_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1], matmul_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0], matmul_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K], matmul_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K], matmul_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K], matmul_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K], matmul_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K], matmul_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S],   matmul_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M],   matmul_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S],   matmul_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S],   matmul_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4],   matmul_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4],   matmul_nvfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        } else {
            CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0].f32acc, matmul_q1_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );

            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S].f32acc,   matmul_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M].f32acc,   matmul_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS].f32acc, matmul_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS].f32acc,  matmul_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S].f32acc,   matmul_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS].f32acc, matmul_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S].f32acc,   matmul_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS].f32acc,  matmul_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL].f32acc,  matmul_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4].f32acc,   matmul_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
            CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4].f32acc,   matmul_nvfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, );
        }

        GGML_ASSERT(device->subgroup_ballot);

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_subgroup_f32_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_subgroup_f16_f32, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (device->coopmat_bf16_support) {
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        }
#endif

        CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_subgroup_q1_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_subgroup_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_subgroup_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_subgroup_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_subgroup_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_subgroup_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_subgroup_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_subgroup_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_subgroup_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_subgroup_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_subgroup_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
        CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_subgroup_nvfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id);
#undef CREATE_MM2
#undef CREATE_MM
    } else
#endif  // defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
    if (device->fp16) {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
        // Selects dot2 SPIR-V variant at runtime when device->dot2_f16 is true
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", (device->dot2_f16 ? NAMELC ## _dot2 ## F16ACC ## _len : NAMELC ## F16ACC ## _len), (device->dot2_f16 ? NAMELC ## _dot2 ## F16ACC ## _data : NAMELC ## F16ACC ## _data), "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", (device->dot2_f16 ? NAMELC ## _dot2 ## F16ACC ## _len : NAMELC ## F16ACC ## _len), (device->dot2_f16 ? NAMELC ## _dot2 ## F16ACC ## _data : NAMELC ## F16ACC ## _data), "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", (device->dot2_f16 ? NAMELC ## _dot2 ## F16ACC ## _len : NAMELC ## F16ACC ## _len), (device->dot2_f16 ? NAMELC ## _dot2 ## F16ACC ## _data : NAMELC ## F16ACC ## _data), "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", (device->dot2_f16 ? NAMELC ## _dot2_aligned ## F16ACC ## _len : NAMELC ## _aligned ## F16ACC ## _len), (device->dot2_f16 ? NAMELC ## _dot2_aligned ## F16ACC ## _data : NAMELC ## _aligned ## F16ACC ## _data), "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", (device->dot2_f16 ? NAMELC ## _dot2_aligned ## F16ACC ## _len : NAMELC ## _aligned ## F16ACC ## _len), (device->dot2_f16 ? NAMELC ## _dot2_aligned ## F16ACC ## _data : NAMELC ## _aligned ## F16ACC ## _data), "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", (device->dot2_f16 ? NAMELC ## _dot2_aligned ## F16ACC ## _len : NAMELC ## _aligned ## F16ACC ## _len), (device->dot2_f16 ? NAMELC ## _dot2_aligned ## F16ACC ## _data : NAMELC ## _aligned ## F16ACC ## _data), "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \

        // bf16 scalar path promotes to f32, no dot2 variant
#define CREATE_MM_NODOT2(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _len, NAMELC ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _len, NAMELC ## _aligned ## F16ACC ## _data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l_int[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->l, #NAMELC        "_l", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \
        if (device->mul_mat ## ID ## _m_int[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->m, #NAMELC        "_m", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \
        if (device->mul_mat ## ID ## _s_int[TYPE]) { \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME .f32acc->s, #NAMELC        "_s", NAMELC ## _len,        NAMELC ##  _data,        "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        } \

        // Create 2 variants, {f16,f32} accumulator
#define CREATE_MM2(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        CREATE_MM(TYPE, PIPELINE_NAME . f16acc, NAMELC, _f16acc, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        CREATE_MM(TYPE, PIPELINE_NAME . f32acc, NAMELC, , WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16, matmul_f16, wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_f16_f32, matmul_f16_f32, wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM_NODOT2(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0], matmul_q1_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0], matmul_q4_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1], matmul_q4_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0], matmul_q5_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1], matmul_q5_1_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0], matmul_q8_0_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K], matmul_q2_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K], matmul_q3_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K], matmul_q4_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K], matmul_q5_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K], matmul_q6_k_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S],   matmul_iq1_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M],   matmul_iq1_m_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS], matmul_iq2_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS],  matmul_iq2_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S],   matmul_iq2_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS], matmul_iq3_xxs_f32, mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S],   matmul_iq3_s_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS],  matmul_iq4_xs_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL],  matmul_iq4_nl_f32,  mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4],   matmul_mxfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4],   matmul_nvfp4_f32,   mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_0], matmul_q4_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_1], matmul_q4_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_0], matmul_q5_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_1], matmul_q5_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q8_0], matmul_q8_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);

            CREATE_MMQ(GGML_TYPE_MXFP4, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_MXFP4], matmul_mxfp4_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, , 0);

            CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q2_K], matmul_q2_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q3_K], matmul_q3_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_K], matmul_q4_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_K], matmul_q5_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
            CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q6_K], matmul_q6_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, , 0);
        }
#endif

        if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_subgroup_f32_f32, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_subgroup_f16, wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_subgroup_f16_f32, wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM_NODOT2(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_subgroup_q1_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_subgroup_iq1_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_subgroup_iq1_m_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_subgroup_iq2_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_subgroup_iq2_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_subgroup_iq2_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_subgroup_iq3_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_subgroup_iq3_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_subgroup_iq4_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_subgroup_iq4_nl_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_subgroup_mxfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_subgroup_nvfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_0], matmul_id_subgroup_q4_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_1], matmul_id_subgroup_q4_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_0], matmul_id_subgroup_q5_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_1], matmul_id_subgroup_q5_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
                CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q8_0], matmul_id_subgroup_q8_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);

                CREATE_MMQ(GGML_TYPE_MXFP4, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_MXFP4], matmul_id_subgroup_mxfp4_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);

                CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q2_K], matmul_id_subgroup_q2_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q3_K], matmul_id_subgroup_q3_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_K], matmul_id_subgroup_q4_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_K], matmul_id_subgroup_q5_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
                CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q6_K], matmul_id_subgroup_q6_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            }
#endif
        } else {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16, matmul_id_f16, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_F16, pipeline_matmul_id_f16_f32, matmul_id_f16_f32, wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM_NODOT2(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0], matmul_id_q1_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0], matmul_id_q4_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1], matmul_id_q4_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0], matmul_id_q5_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1], matmul_id_q5_1_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0], matmul_id_q8_0_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K], matmul_id_q2_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K], matmul_id_q3_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K], matmul_id_q4_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K], matmul_id_q5_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K], matmul_id_q6_k_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S],   matmul_id_iq1_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M],   matmul_id_iq1_m_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS], matmul_id_iq2_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS],  matmul_id_iq2_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S],   matmul_id_iq2_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS], matmul_id_iq3_xxs_f32, mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S],   matmul_id_iq3_s_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS],  matmul_id_iq4_xs_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL],  matmul_id_iq4_nl_f32,  mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4],   matmul_id_mxfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM2(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4],   matmul_id_nvfp4_f32,   mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_0], matmul_id_q4_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_1], matmul_id_q4_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_0], matmul_id_q5_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_1], matmul_id_q5_1_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q8_0], matmul_id_q8_0_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

                CREATE_MMQ(GGML_TYPE_MXFP4, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_MXFP4], matmul_id_mxfp4_q8_1, mmq_wg_denoms, warptile_mmqid_int,   vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

                CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q2_K], matmul_id_q2_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q3_K], matmul_id_q3_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q4_K], matmul_id_q4_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q5_K], matmul_id_q5_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
                CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id_q8_1[GGML_TYPE_Q6_K], matmul_id_q6_k_q8_1, mmq_wg_denoms, warptile_mmqid_int_k, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            }
#endif
        }
#undef CREATE_MM2
#undef CREATE_MMQ
#undef CREATE_MM
#undef CREATE_MM_NODOT2
    } else {
        // Create 6 variants, {s,m,l}x{unaligned,aligned}
#define CREATE_MM(TYPE, PIPELINE_NAME, NAMELC, F16ACC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID, REQSUBGROUPSIZE) \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC #F16ACC "_l", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC #F16ACC "_m", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC #F16ACC "_s", NAMELC ## F16ACC ## _fp32_len, NAMELC ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _l[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_l, #NAMELC #F16ACC "_aligned_l", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, l_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _m[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_m, #NAMELC #F16ACC "_aligned_m", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, m_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \
        if (device->mul_mat ## ID ## _s[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->a_s, #NAMELC #F16ACC "_aligned_s", NAMELC ## _aligned ## F16ACC ## _fp32_len, NAMELC ## _aligned ## F16ACC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, s_align, false, REQSUBGROUPSIZE > 0, REQSUBGROUPSIZE);   \

#define CREATE_MMQ(TYPE, PIPELINE_NAME, NAMELC, WG_DENOMS, WARPTILE, PUSHCONST, PARAMCOUNT, ID) \
        if (device->mul_mat ## ID ## _l_int[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->l, #NAMELC "_l", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), l_ ## WG_DENOMS, l_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _m_int[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->m, #NAMELC "_m", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), m_ ## WG_DENOMS, m_ ## WARPTILE, 1);   \
        if (device->mul_mat ## ID ## _s_int[TYPE]) \
            ggml_vk_create_pipeline(device, device-> PIPELINE_NAME ->s, #NAMELC "_s", NAMELC ## _fp32_len, NAMELC ## _fp32_data, "main", PARAMCOUNT, sizeof(PUSHCONST), s_ ## WG_DENOMS, s_ ## WARPTILE, 1);   \

        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32, matmul_f32_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F32, pipeline_matmul_f32_f16, matmul_f32_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_f16.f32acc, matmul_f16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_F16, pipeline_matmul_f16_f32.f32acc, matmul_f16_f32, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q1_0].f32acc, matmul_q1_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

        CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_S].f32acc,   matmul_iq1_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ1_M].f32acc,   matmul_iq1_m_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XXS].f32acc, matmul_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_XS].f32acc,  matmul_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ2_S].f32acc,   matmul_iq2_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_XXS].f32acc, matmul_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ3_S].f32acc,   matmul_iq3_s_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_XS].f32acc,  matmul_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat[GGML_TYPE_IQ4_NL].f32acc,  matmul_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_MXFP4].f32acc,   matmul_mxfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat[GGML_TYPE_NVFP4].f32acc,   matmul_nvfp4_f32,   , mmq_wg_denoms, warptile_mmq, vk_mat_mat_push_constants, 3, , 0);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            CREATE_MMQ(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_0].f32acc, matmul_q4_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_1].f32acc, matmul_q4_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_0].f32acc, matmul_q5_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_1].f32acc, matmul_q5_1_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q8_0].f32acc, matmul_q8_0_q8_1, mmq_wg_denoms, warptile_mmq_int, vk_mat_mat_push_constants, 3, );

            CREATE_MMQ(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q2_K].f32acc, matmul_q2_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q3_K].f32acc, matmul_q3_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q4_K].f32acc, matmul_q4_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q5_K].f32acc, matmul_q5_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
            CREATE_MMQ(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_q8_1[GGML_TYPE_Q6_K].f32acc, matmul_q6_k_q8_1, mmq_wg_denoms, warptile_mmq_int_k, vk_mat_mat_push_constants, 3, );
        }
#endif

        if (device->subgroup_ballot && device->subgroup_require_full_support && subgroup_min_size_16) {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_subgroup_f32_f32, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16.f32acc, matmul_id_subgroup_f16, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16_f32.f32acc, matmul_id_subgroup_f16_f32, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_subgroup_bf16, , wg_denoms, warptile_id, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size_16);

            CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0].f32acc, matmul_id_subgroup_q1_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f32acc, matmul_id_subgroup_q4_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f32acc, matmul_id_subgroup_q4_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f32acc, matmul_id_subgroup_q5_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f32acc, matmul_id_subgroup_q5_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f32acc, matmul_id_subgroup_q8_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f32acc, matmul_id_subgroup_q2_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f32acc, matmul_id_subgroup_q3_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f32acc, matmul_id_subgroup_q4_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f32acc, matmul_id_subgroup_q5_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f32acc, matmul_id_subgroup_q6_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f32acc,   matmul_id_subgroup_iq1_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f32acc,   matmul_id_subgroup_iq1_m_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f32acc, matmul_id_subgroup_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f32acc,  matmul_id_subgroup_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f32acc,   matmul_id_subgroup_iq2_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f32acc, matmul_id_subgroup_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f32acc,   matmul_id_subgroup_iq3_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f32acc,  matmul_id_subgroup_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f32acc,  matmul_id_subgroup_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f32acc,   matmul_id_subgroup_mxfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
            CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4].f32acc,   matmul_id_subgroup_nvfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, mul_mat_subgroup_size);
        } else {
            CREATE_MM(GGML_TYPE_F32, pipeline_matmul_id_f32, matmul_id_f32_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16.f32acc, matmul_id_f16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_F16, pipeline_matmul_id_f16_f32.f32acc, matmul_id_f16_f32, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);

            CREATE_MM(GGML_TYPE_Q1_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q1_0].f32acc, matmul_id_q1_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q4_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_0].f32acc, matmul_id_q4_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q4_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_1].f32acc, matmul_id_q4_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q5_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_0].f32acc, matmul_id_q5_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q5_1, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_1].f32acc, matmul_id_q5_1_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q8_0, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q8_0].f32acc, matmul_id_q8_0_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q2_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q2_K].f32acc, matmul_id_q2_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q3_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q3_K].f32acc, matmul_id_q3_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q4_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q4_K].f32acc, matmul_id_q4_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q5_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q5_K].f32acc, matmul_id_q5_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_Q6_K, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_Q6_K].f32acc, matmul_id_q6_k_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ1_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_S].f32acc,   matmul_id_iq1_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ1_M,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ1_M].f32acc,   matmul_id_iq1_m_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ2_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XXS].f32acc, matmul_id_iq2_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ2_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_XS].f32acc,  matmul_id_iq2_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ2_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ2_S].f32acc,   matmul_id_iq2_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ3_XXS, pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_XXS].f32acc, matmul_id_iq3_xxs_f32, , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ3_S,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ3_S].f32acc,   matmul_id_iq3_s_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ4_XS,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_XS].f32acc,  matmul_id_iq4_xs_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_IQ4_NL,  pipeline_dequant_mul_mat_mat_id[GGML_TYPE_IQ4_NL].f32acc,  matmul_id_iq4_nl_f32,  , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_MXFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_MXFP4].f32acc,   matmul_id_mxfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
            CREATE_MM(GGML_TYPE_NVFP4,   pipeline_dequant_mul_mat_mat_id[GGML_TYPE_NVFP4].f32acc,   matmul_id_nvfp4_f32,   , mmq_wg_denoms, warptile_mmqid, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
        }
    }
    // reusing CREATE_MM from the fp32 path
    if ((device->coopmat2 || device->coopmat_support)
#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        && !device->coopmat_bf16_support
#endif
        ) {
        const uint32_t s_warptile_wm = device->subgroup_size == 8 ? 8 : 32;

        // use scalar tile sizes
        l_warptile = { 128, 128, 128, 16, subgroup_size_8 * 2, 64, 2, 4, 4, 1, subgroup_size_8 };
        m_warptile = { 128,  64,  64, 16, subgroup_size_8, 32, 2, 4, 2, 1, subgroup_size_8 };
        s_warptile = { subgroup_size_32, 32, 32, 16, s_warptile_wm, 32, 2, 2, 2, 1, subgroup_size_8 };

        l_wg_denoms = {128, 128, 1 };
        m_wg_denoms = { 64,  64, 1 };
        s_wg_denoms = { 32,  32, 1 };

        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_bf16, matmul_bf16, , wg_denoms, warptile, vk_mat_mat_push_constants, 3, , 0);
        CREATE_MM(GGML_TYPE_BF16, pipeline_matmul_id_bf16, matmul_id_bf16, , wg_denoms, warptile, vk_mat_mat_id_push_constants, mul_mat_id_param_count, _id, 0);
    }
#undef CREATE_MM

    // mul mat vec

    // the number of rows computed per shader depends on GPU model and quant
    uint32_t rm_stdq = 1;
    uint32_t rm_kq = 2;
    uint32_t rm_stdq_int = 1;
    uint32_t rm_kq_int = 1;
    auto const &rm_iq_int = [](uint32_t i) { return i == 0 ? 8u : 4u; };
    if (device->vendor_id == VK_VENDOR_ID_AMD) {
        if (device->architecture == AMD_GCN) {
            rm_stdq = 2;
            rm_kq = 4;
            rm_stdq_int = 4;
        }
    } else if (device->vendor_id == VK_VENDOR_ID_INTEL) {
        rm_stdq = 2;
        rm_stdq_int = 2;
    }
    uint32_t rm_iq = 2 * rm_kq;

    const bool use_subgroups = device->subgroup_arithmetic && device->architecture != vk_device_architecture::AMD_GCN;
    // Ensure a subgroup size >= 16 is available
    const bool use_subgroups16 = use_subgroups && subgroup_min_size_16;

    const uint32_t subgroup_size = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control && device->subgroup_min_size <= 16 && device->subgroup_max_size >= 16) ? 16 : device->subgroup_size;
    const uint32_t subgroup_size16 = std::max(subgroup_size, 16u);

    const uint32_t force_subgroup_size = use_subgroups ? subgroup_size : 0;
    const uint32_t force_subgroup_size16 = use_subgroups16 ? subgroup_size16 : 0;
    static constexpr uint32_t mul_mat_vec_num_bindings = 5;
    static constexpr uint32_t mul_mat_vec_id_num_bindings = 6;

    for (uint32_t w = 0; w < DMMV_WG_SIZE_COUNT; ++w) {
        const uint32_t wg_size_subgroup   = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size : (subgroup_size * 4);
        const uint32_t wg_size_subgroup16 = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);

        const shader_reduction_mode reduc = (use_subgroups && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                            (use_subgroups && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                            SHADER_REDUCTION_MODE_SHMEM;

        const shader_reduction_mode reduc16 = (use_subgroups16 && w == DMMV_WG_SIZE_SUBGROUP) ? SHADER_REDUCTION_MODE_SUBGROUP :
                                              (use_subgroups16 && w == DMMV_WG_SIZE_LARGE) ? SHADER_REDUCTION_MODE_HYBRID :
                                              SHADER_REDUCTION_MODE_SHMEM;

        for (uint32_t i = 0; i < mul_mat_vec_max_cols; ++i) {
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f32_f32",  arr_dmmv_f32_f32_f32_len[reduc],  arr_dmmv_f32_f32_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f32_f32",  arr_dmmv_f16_f32_f32_len[reduc],  arr_dmmv_f16_f32_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f32_f32", arr_dmmv_bf16_f32_f32_len[reduc], arr_dmmv_bf16_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f32_f32", arr_dmmv_q1_0_f32_f32_len[reduc], arr_dmmv_q1_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f32_f32", arr_dmmv_q4_0_f32_f32_len[reduc], arr_dmmv_q4_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f32_f32", arr_dmmv_q4_1_f32_f32_len[reduc], arr_dmmv_q4_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f32_f32", arr_dmmv_q5_0_f32_f32_len[reduc], arr_dmmv_q5_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f32_f32", arr_dmmv_q5_1_f32_f32_len[reduc], arr_dmmv_q5_1_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f32_f32", arr_dmmv_q8_0_f32_f32_len[reduc], arr_dmmv_q8_0_f32_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f32_f32", arr_dmmv_q2_k_f32_f32_len[reduc16], arr_dmmv_q2_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f32_f32", arr_dmmv_q3_k_f32_f32_len[reduc16], arr_dmmv_q3_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f32_f32", arr_dmmv_q4_k_f32_f32_len[reduc16], arr_dmmv_q4_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f32_f32", arr_dmmv_q5_k_f32_f32_len[reduc16], arr_dmmv_q5_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f32_f32", arr_dmmv_q6_k_f32_f32_len[reduc16], arr_dmmv_q6_k_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f32_f32",   arr_dmmv_iq1_s_f32_f32_len[reduc16],   arr_dmmv_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f32_f32",   arr_dmmv_iq1_m_f32_f32_len[reduc16],   arr_dmmv_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f32_f32", arr_dmmv_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f32_f32",  arr_dmmv_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f32_f32",   arr_dmmv_iq2_s_f32_f32_len[reduc16],   arr_dmmv_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f32_f32", arr_dmmv_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f32_f32",   arr_dmmv_iq3_s_f32_f32_len[reduc16],   arr_dmmv_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f32_f32",  arr_dmmv_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f32_f32",  arr_dmmv_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f32_f32",   arr_dmmv_mxfp4_f32_f32_len[reduc16],   arr_dmmv_mxfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f32_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f32_f32",   arr_dmmv_nvfp4_f32_f32_len[reduc16],   arr_dmmv_nvfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F32 ][i], "mul_mat_vec_f32_f16_f32",  arr_dmmv_f32_f16_f32_len[reduc],  arr_dmmv_f32_f16_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1, 1, 1}, {wg_size_subgroup, 1, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_F16 ][i], "mul_mat_vec_f16_f16_f32",  arr_dmmv_f16_f16_f32_len[reduc],  arr_dmmv_f16_f16_f32_data[reduc],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_BF16][i], "mul_mat_vec_bf16_f16_f32", arr_dmmv_bf16_f16_f32_len[reduc], arr_dmmv_bf16_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2, 1, 1}, {wg_size_subgroup, 2, i+1}, 1, false, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q1_0][i], "mul_mat_vec_q1_0_f16_f32", arr_dmmv_q1_0_f16_f32_len[reduc], arr_dmmv_q1_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_f16_f32", arr_dmmv_q4_0_f16_f32_len[reduc], arr_dmmv_q4_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_f16_f32", arr_dmmv_q4_1_f16_f32_len[reduc], arr_dmmv_q4_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_f16_f32", arr_dmmv_q5_0_f16_f32_len[reduc], arr_dmmv_q5_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_f16_f32", arr_dmmv_q5_1_f16_f32_len[reduc], arr_dmmv_q5_1_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_f16_f32", arr_dmmv_q8_0_f16_f32_len[reduc], arr_dmmv_q8_0_f16_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq, i+1}, 1, true, use_subgroups, force_subgroup_size);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_f16_f32", arr_dmmv_q2_k_f16_f32_len[reduc16], arr_dmmv_q2_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_f16_f32", arr_dmmv_q3_k_f16_f32_len[reduc16], arr_dmmv_q3_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_f16_f32", arr_dmmv_q4_k_f16_f32_len[reduc16], arr_dmmv_q4_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_f16_f32", arr_dmmv_q5_k_f16_f32_len[reduc16], arr_dmmv_q5_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_f16_f32", arr_dmmv_q6_k_f16_f32_len[reduc16], arr_dmmv_q6_k_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_S][i],   "mul_mat_vec_iq1_s_f16_f32",   arr_dmmv_iq1_s_f16_f32_len[reduc16],   arr_dmmv_iq1_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ1_M][i],   "mul_mat_vec_iq1_m_f16_f32",   arr_dmmv_iq1_m_f16_f32_len[reduc16],   arr_dmmv_iq1_m_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XXS][i], "mul_mat_vec_iq2_xxs_f16_f32", arr_dmmv_iq2_xxs_f16_f32_len[reduc16], arr_dmmv_iq2_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_XS][i],  "mul_mat_vec_iq2_xs_f16_f32",  arr_dmmv_iq2_xs_f16_f32_len[reduc16],  arr_dmmv_iq2_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ2_S][i],   "mul_mat_vec_iq2_s_f16_f32",   arr_dmmv_iq2_s_f16_f32_len[reduc16],   arr_dmmv_iq2_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_XXS][i], "mul_mat_vec_iq3_xxs_f16_f32", arr_dmmv_iq3_xxs_f16_f32_len[reduc16], arr_dmmv_iq3_xxs_f16_f32_data[reduc16], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ3_S][i],   "mul_mat_vec_iq3_s_f16_f32",   arr_dmmv_iq3_s_f16_f32_len[reduc16],   arr_dmmv_iq3_s_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_XS][i],  "mul_mat_vec_iq4_xs_f16_f32",  arr_dmmv_iq4_xs_f16_f32_len[reduc16],  arr_dmmv_iq4_xs_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_IQ4_NL][i],  "mul_mat_vec_iq4_nl_f16_f32",  arr_dmmv_iq4_nl_f16_f32_len[reduc16],  arr_dmmv_iq4_nl_f16_f32_data[reduc16],  "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_MXFP4][i],   "mul_mat_vec_mxfp4_f16_f32",   arr_dmmv_mxfp4_f16_f32_len[reduc16],   arr_dmmv_mxfp4_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_f16_f32[w][GGML_TYPE_NVFP4][i],   "mul_mat_vec_nvfp4_f16_f32",   arr_dmmv_nvfp4_f16_f32_len[reduc16],   arr_dmmv_nvfp4_f16_f32_data[reduc16],   "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq, i+1}, 1, true, use_subgroups16, force_subgroup_size16);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
            if (device->integer_dot_product) {
                const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
                const uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32", arr_dmmv_q4_0_q8_1_f32_len[reduc], arr_dmmv_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_1][i], "mul_mat_vec_q4_1_q8_1_f32", arr_dmmv_q4_1_q8_1_f32_len[reduc], arr_dmmv_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_0][i], "mul_mat_vec_q5_0_q8_1_f32", arr_dmmv_q5_0_q8_1_f32_len[reduc], arr_dmmv_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_1][i], "mul_mat_vec_q5_1_q8_1_f32", arr_dmmv_q5_1_q8_1_f32_len[reduc], arr_dmmv_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q8_0][i], "mul_mat_vec_q8_0_q8_1_f32", arr_dmmv_q8_0_q8_1_f32_len[reduc], arr_dmmv_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_MXFP4][i], "mul_mat_vec_mxfp4_q8_1_f32", arr_dmmv_mxfp4_q8_1_f32_len[reduc], arr_dmmv_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_stdq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q2_K][i], "mul_mat_vec_q2_k_q8_1_f32", arr_dmmv_q2_k_q8_1_f32_len[reduc], arr_dmmv_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {2*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q3_K][i], "mul_mat_vec_q3_k_q8_1_f32", arr_dmmv_q3_k_q8_1_f32_len[reduc], arr_dmmv_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_K][i], "mul_mat_vec_q4_k_q8_1_f32", arr_dmmv_q4_k_q8_1_f32_len[reduc], arr_dmmv_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q5_K][i], "mul_mat_vec_q5_k_q8_1_f32", arr_dmmv_q5_k_q8_1_f32_len[reduc], arr_dmmv_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q6_K][i], "mul_mat_vec_q6_k_q8_1_f32", arr_dmmv_q6_k_q8_1_f32_len[reduc], arr_dmmv_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int, i+1}, 1, true, use_subgroups, subgroup_size_int);

                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_S][i], "mul_mat_vec_iq1_s_q8_1_f32", arr_dmmv_iq1_s_q8_1_f32_len[reduc], arr_dmmv_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_size_int);
                ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_IQ1_M][i], "mul_mat_vec_iq1_m_q8_1_f32", arr_dmmv_iq1_m_q8_1_f32_len[reduc], arr_dmmv_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_push_constants), {1*rm_iq_int(i), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(i), i+1}, 1, true, use_subgroups, subgroup_size_int);

            }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
        }

        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F32 ], "mul_mat_vec_id_f32_f32",        arr_dmmv_id_f32_f32_f32_len[reduc],     arr_dmmv_id_f32_f32_f32_data[reduc],     "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1, 1, 1}, {wg_size_subgroup, 1}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_F16 ], "mul_mat_vec_id_f16_f32",        arr_dmmv_id_f16_f32_f32_len[reduc],     arr_dmmv_id_f16_f32_f32_data[reduc],     "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_BF16], "mul_mat_vec_id_bf16_f32",       arr_dmmv_id_bf16_f32_f32_len[reduc],    arr_dmmv_id_bf16_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2, 1, 1}, {wg_size_subgroup, 2}, 1, false, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q1_0], "mul_mat_vec_id_q1_0_f32",       arr_dmmv_id_q1_0_f32_f32_len[reduc],    arr_dmmv_id_q1_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_f32",       arr_dmmv_id_q4_0_f32_f32_len[reduc],    arr_dmmv_id_q4_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_f32",       arr_dmmv_id_q4_1_f32_f32_len[reduc],    arr_dmmv_id_q4_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_f32",       arr_dmmv_id_q5_0_f32_f32_len[reduc],    arr_dmmv_id_q5_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_f32",       arr_dmmv_id_q5_1_f32_f32_len[reduc],    arr_dmmv_id_q5_1_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq, 1, 1}, {wg_size_subgroup, 2*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_f32",       arr_dmmv_id_q8_0_f32_f32_len[reduc],    arr_dmmv_id_q8_0_f32_f32_data[reduc],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq, 1, 1}, {wg_size_subgroup, 1*rm_stdq}, 1, true, use_subgroups, force_subgroup_size);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_f32",       arr_dmmv_id_q2_k_f32_f32_len[reduc16],    arr_dmmv_id_q2_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_f32",       arr_dmmv_id_q3_k_f32_f32_len[reduc16],    arr_dmmv_id_q3_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_f32",       arr_dmmv_id_q4_k_f32_f32_len[reduc16],    arr_dmmv_id_q4_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_f32",       arr_dmmv_id_q5_k_f32_f32_len[reduc16],    arr_dmmv_id_q5_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_f32",       arr_dmmv_id_q6_k_f32_f32_len[reduc16],    arr_dmmv_id_q6_k_f32_f32_data[reduc16],    "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_kq, 1, 1}, {wg_size_subgroup16, rm_kq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_S],   "mul_mat_vec_id_iq1_s_f32",   arr_dmmv_id_iq1_s_f32_f32_len[reduc16],   arr_dmmv_id_iq1_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ1_M],   "mul_mat_vec_id_iq1_m_f32",   arr_dmmv_id_iq1_m_f32_f32_len[reduc16],   arr_dmmv_id_iq1_m_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XXS], "mul_mat_vec_id_iq2_xxs_f32", arr_dmmv_id_iq2_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq2_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_XS],  "mul_mat_vec_id_iq2_xs_f32",  arr_dmmv_id_iq2_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq2_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ2_S],   "mul_mat_vec_id_iq2_s_f32",   arr_dmmv_id_iq2_s_f32_f32_len[reduc16],   arr_dmmv_id_iq2_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_XXS], "mul_mat_vec_id_iq3_xxs_f32", arr_dmmv_id_iq3_xxs_f32_f32_len[reduc16], arr_dmmv_id_iq3_xxs_f32_f32_data[reduc16], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ3_S],   "mul_mat_vec_id_iq3_s_f32",   arr_dmmv_id_iq3_s_f32_f32_len[reduc16],   arr_dmmv_id_iq3_s_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_XS],  "mul_mat_vec_id_iq4_xs_f32",  arr_dmmv_id_iq4_xs_f32_f32_len[reduc16],  arr_dmmv_id_iq4_xs_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_IQ4_NL],  "mul_mat_vec_id_iq4_nl_f32",  arr_dmmv_id_iq4_nl_f32_f32_len[reduc16],  arr_dmmv_id_iq4_nl_f32_f32_data[reduc16],  "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_MXFP4],   "mul_mat_vec_id_mxfp4_f32",   arr_dmmv_id_mxfp4_f32_f32_len[reduc16],   arr_dmmv_id_mxfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);
        ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_f32[w][GGML_TYPE_NVFP4],   "mul_mat_vec_id_nvfp4_f32",   arr_dmmv_id_nvfp4_f32_f32_len[reduc16],   arr_dmmv_id_nvfp4_f32_f32_data[reduc16],   "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {rm_iq, 1, 1}, {wg_size_subgroup16, rm_iq}, 1, true, use_subgroups16, force_subgroup_size16);

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (device->integer_dot_product) {
            const uint32_t subgroup_size_int = (device->vendor_id == VK_VENDOR_ID_INTEL && device->subgroup_size_control) ? device->subgroup_min_size : device->subgroup_size;
            const uint32_t wg_size_subgroup_int = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size_int : (subgroup_size_int * 4);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32", arr_dmmv_id_q4_0_q8_1_f32_len[reduc], arr_dmmv_id_q4_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_1], "mul_mat_vec_id_q4_1_q8_1_f32", arr_dmmv_id_q4_1_q8_1_f32_len[reduc], arr_dmmv_id_q4_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_0], "mul_mat_vec_id_q5_0_q8_1_f32", arr_dmmv_id_q5_0_q8_1_f32_len[reduc], arr_dmmv_id_q5_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_1], "mul_mat_vec_id_q5_1_q8_1_f32", arr_dmmv_id_q5_1_q8_1_f32_len[reduc], arr_dmmv_id_q5_1_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q8_0], "mul_mat_vec_id_q8_0_q8_1_f32", arr_dmmv_id_q8_0_q8_1_f32_len[reduc], arr_dmmv_id_q8_0_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_MXFP4], "mul_mat_vec_id_mxfp4_q8_1_f32", arr_dmmv_id_mxfp4_q8_1_f32_len[reduc], arr_dmmv_id_mxfp4_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_stdq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_stdq_int}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q2_K], "mul_mat_vec_id_q2_k_q8_1_f32", arr_dmmv_id_q2_k_q8_1_f32_len[reduc], arr_dmmv_id_q2_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {2*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 2*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q3_K], "mul_mat_vec_id_q3_k_q8_1_f32", arr_dmmv_id_q3_k_q8_1_f32_len[reduc], arr_dmmv_id_q3_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_K], "mul_mat_vec_id_q4_k_q8_1_f32", arr_dmmv_id_q4_k_q8_1_f32_len[reduc], arr_dmmv_id_q4_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q5_K], "mul_mat_vec_id_q5_k_q8_1_f32", arr_dmmv_id_q5_k_q8_1_f32_len[reduc], arr_dmmv_id_q5_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q6_K], "mul_mat_vec_id_q6_k_q8_1_f32", arr_dmmv_id_q6_k_q8_1_f32_len[reduc], arr_dmmv_id_q6_k_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_kq_int, 1, 1}, {wg_size_subgroup_int, 1*rm_kq_int}, 1, true, use_subgroups, subgroup_size_int);

            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_S], "mul_mat_vec_id_iq1_s_q8_1_f32", arr_dmmv_id_iq1_s_q8_1_f32_len[reduc], arr_dmmv_id_iq1_s_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_size_int);
            ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_IQ1_M], "mul_mat_vec_id_iq1_m_q8_1_f32", arr_dmmv_id_iq1_m_q8_1_f32_len[reduc], arr_dmmv_id_iq1_m_q8_1_f32_data[reduc], "main", mul_mat_vec_id_num_bindings, sizeof(vk_mat_vec_id_push_constants), {1*rm_iq_int(0), 1, 1}, {wg_size_subgroup_int, 1*rm_iq_int(0)}, 1, true, use_subgroups, subgroup_size_int);
        }
#endif // GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
    }

#if !defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    GGML_UNUSED(rm_stdq_int);
    GGML_UNUSED(rm_kq_int);
    GGML_UNUSED(rm_iq_int);
#endif

    // dequant shaders
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_F32 ], "f32_to_f16",   dequant_f32_len,  dequant_f32_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q1_0], "dequant_q1_0", dequant_q1_0_len, dequant_q1_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 8, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_0], "dequant_q4_0", dequant_q4_0_len, dequant_q4_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_1], "dequant_q4_1", dequant_q4_1_len, dequant_q4_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_0], "dequant_q5_0", dequant_q5_0_len, dequant_q5_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_1], "dequant_q5_1", dequant_q5_1_len, dequant_q5_1_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q8_0], "dequant_q8_0", dequant_q8_0_len, dequant_q8_0_data, "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q2_K], "dequant_q2_k", dequant_q2_k_len, dequant_q2_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q3_K], "dequant_q3_k", dequant_q3_k_len, dequant_q3_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q4_K], "dequant_q4_k", dequant_q4_k_len, dequant_q4_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q5_K], "dequant_q5_k", dequant_q5_k_len, dequant_q5_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_Q6_K], "dequant_q6_k", dequant_q6_k_len, dequant_q6_k_data, "main", 2, 5 * sizeof(uint32_t), {256 * 64, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_S],   "dequant_iq1_s",   dequant_iq1_s_len,   dequant_iq1_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ1_M],   "dequant_iq1_m",   dequant_iq1_m_len,   dequant_iq1_m_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XXS], "dequant_iq2_xxs", dequant_iq2_xxs_len, dequant_iq2_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_XS],  "dequant_iq2_xs",  dequant_iq2_xs_len,  dequant_iq2_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ2_S],   "dequant_iq2_s",   dequant_iq2_s_len,   dequant_iq2_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_XXS], "dequant_iq3_xxs", dequant_iq3_xxs_len, dequant_iq3_xxs_data, "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ3_S],   "dequant_iq3_s",   dequant_iq3_s_len,   dequant_iq3_s_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_XS],  "dequant_iq4_xs",  dequant_iq4_xs_len,  dequant_iq4_xs_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_IQ4_NL],  "dequant_iq4_nl",  dequant_iq4_nl_len,  dequant_iq4_nl_data,  "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_MXFP4],   "dequant_mxfp4",   dequant_mxfp4_len,   dequant_mxfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_dequant[GGML_TYPE_NVFP4],   "dequant_nvfp4",   dequant_nvfp4_len,   dequant_nvfp4_data,   "main", 2, 5 * sizeof(uint32_t), {256 * 16, 1, 1}, {}, 1);

    // get_rows
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F32 ], "get_rows_f32",  get_rows_f32_len,  get_rows_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_F16 ], "get_rows_f16",  get_rows_f16_len,  get_rows_f16_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_BF16], "get_rows_bf16", get_rows_bf16_len, get_rows_bf16_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q1_0], "get_rows_q1_0", get_rows_q1_0_len, get_rows_q1_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_0], "get_rows_q4_0", get_rows_q4_0_len, get_rows_q4_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_1], "get_rows_q4_1", get_rows_q4_1_len, get_rows_q4_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_0], "get_rows_q5_0", get_rows_q5_0_len, get_rows_q5_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_1], "get_rows_q5_1", get_rows_q5_1_len, get_rows_q5_1_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q8_0], "get_rows_q8_0", get_rows_q8_0_len, get_rows_q8_0_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q2_K], "get_rows_q2_k", get_rows_q2_k_len, get_rows_q2_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q3_K], "get_rows_q3_k", get_rows_q3_k_len, get_rows_q3_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q4_K], "get_rows_q4_k", get_rows_q4_k_len, get_rows_q4_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q5_K], "get_rows_q5_k", get_rows_q5_k_len, get_rows_q5_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_Q6_K], "get_rows_q6_k", get_rows_q6_k_len, get_rows_q6_k_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_S],   "get_rows_iq1_s",   get_rows_iq1_s_len,   get_rows_iq1_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ1_M],   "get_rows_iq1_m",   get_rows_iq1_m_len,   get_rows_iq1_m_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs", get_rows_iq2_xxs_len, get_rows_iq2_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs",  get_rows_iq2_xs_len,  get_rows_iq2_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ2_S],   "get_rows_iq2_s",   get_rows_iq2_s_len,   get_rows_iq2_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs", get_rows_iq3_xxs_len, get_rows_iq3_xxs_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ3_S],   "get_rows_iq3_s",   get_rows_iq3_s_len,   get_rows_iq3_s_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs",  get_rows_iq4_xs_len,  get_rows_iq4_xs_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl",  get_rows_iq4_nl_len,  get_rows_iq4_nl_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_MXFP4],   "get_rows_mxfp4",   get_rows_mxfp4_len,   get_rows_mxfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_NVFP4],   "get_rows_nvfp4",   get_rows_nvfp4_len,   get_rows_nvfp4_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows[GGML_TYPE_I32],     "get_rows_i32",     get_rows_i32_len,     get_rows_i32_data,     "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F32 ], "get_rows_f32_f32",  get_rows_f32_f32_len,  get_rows_f32_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_F16 ], "get_rows_f16_f32",  get_rows_f16_f32_len,  get_rows_f16_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_BF16], "get_rows_bf16_f32", get_rows_bf16_f32_len, get_rows_bf16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), { 512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q1_0], "get_rows_q1_0_f32", get_rows_q1_0_f32_len, get_rows_q1_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_0], "get_rows_q4_0_f32", get_rows_q4_0_f32_len, get_rows_q4_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_1], "get_rows_q4_1_f32", get_rows_q4_1_f32_len, get_rows_q4_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_0], "get_rows_q5_0_f32", get_rows_q5_0_f32_len, get_rows_q5_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_1], "get_rows_q5_1_f32", get_rows_q5_1_f32_len, get_rows_q5_1_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q8_0], "get_rows_q8_0_f32", get_rows_q8_0_f32_len, get_rows_q8_0_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q2_K], "get_rows_q2_k_f32", get_rows_q2_k_f32_len, get_rows_q2_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q3_K], "get_rows_q3_k_f32", get_rows_q3_k_f32_len, get_rows_q3_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q4_K], "get_rows_q4_k_f32", get_rows_q4_k_f32_len, get_rows_q4_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q5_K], "get_rows_q5_k_f32", get_rows_q5_k_f32_len, get_rows_q5_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_Q6_K], "get_rows_q6_k_f32", get_rows_q6_k_f32_len, get_rows_q6_k_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_S],   "get_rows_iq1_s_f32",   get_rows_iq1_s_f32_len,   get_rows_iq1_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ1_M],   "get_rows_iq1_m_f32",   get_rows_iq1_m_f32_len,   get_rows_iq1_m_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XXS], "get_rows_iq2_xxs_f32", get_rows_iq2_xxs_f32_len, get_rows_iq2_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_XS],  "get_rows_iq2_xs_f32",  get_rows_iq2_xs_f32_len,  get_rows_iq2_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ2_S],   "get_rows_iq2_s_f32",   get_rows_iq2_s_f32_len,   get_rows_iq2_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_XXS], "get_rows_iq3_xxs_f32", get_rows_iq3_xxs_f32_len, get_rows_iq3_xxs_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ3_S],   "get_rows_iq3_s_f32",   get_rows_iq3_s_f32_len,   get_rows_iq3_s_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_XS],  "get_rows_iq4_xs_f32",  get_rows_iq4_xs_f32_len,  get_rows_iq4_xs_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_IQ4_NL],  "get_rows_iq4_nl_f32",  get_rows_iq4_nl_f32_len,  get_rows_iq4_nl_f32_data,  "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_MXFP4],   "get_rows_mxfp4_f32",   get_rows_mxfp4_f32_len,   get_rows_mxfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_get_rows_f32[GGML_TYPE_NVFP4],   "get_rows_nvfp4_f32",   get_rows_nvfp4_f32_len,   get_rows_nvfp4_f32_data,   "main", 3, sizeof(vk_op_binary_push_constants), {1024, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_matmul_split_k_reduce, "split_k_reduce", split_k_reduce_len, split_k_reduce_data, "main", 2, 2 * sizeof(uint32_t), {256 * 4, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_flash_attn_split_k_reduce, "fa_split_k_reduce", fa_split_k_reduce_len, fa_split_k_reduce_data, "main", 3, sizeof(vk_op_flash_attn_split_k_reduce_push_constants), {1, device->subgroup_size, 1}, {device->subgroup_size}, 1, true);

    for (auto &it : device->pipeline_fa_mask_opt) {
        auto BrBc = it.first;
        ggml_vk_create_pipeline(device, it.second, "fa_mask_opt", fa_mask_opt_len, fa_mask_opt_data, "main", 2, sizeof(vk_op_flash_attn_mask_opt_push_constants), {1, 1, 1}, {128, 128 / device->subgroup_size, BrBc.first, BrBc.second}, 1, true, true, device->subgroup_size);
    }

    if (device->subgroup_clustered && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_subgroup_len, quantize_q8_1_x4_subgroup_data, "main", 2, sizeof(vk_quantize_q8_1_push_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_quantize_q8_1_x4, "quantize_q8_1_x4", quantize_q8_1_x4_len, quantize_q8_1_x4_data, "main", 2, sizeof(vk_quantize_q8_1_push_constants), {32 * device->subgroup_size / 8, 1, 1}, { device->subgroup_size }, 1);
    }

    for (uint32_t i = 0; i < p021_max_gqa_ratio; ++i) {
        if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_subgroup_add_len, mul_mat_vec_p021_f16_f32_subgroup_add_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_p021_push_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true, true);
        } else {
            ggml_vk_create_pipeline2(device, device->pipeline_mul_mat_vec_p021_f16_f32[i], "mul_mat_vec_p021_f16_f32"+std::to_string(i+1), mul_mat_vec_p021_f16_f32_len,              mul_mat_vec_p021_f16_f32_data,              "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_p021_push_constants), {1, 1, 1}, {device->subgroup_size, i + 1}, 1, true);
        }
    }
    ggml_vk_create_pipeline(device, device->pipeline_mul_mat_vec_nc_f16_f32, "mul_mat_vec_nc_f16_f32", mul_mat_vec_nc_f16_f32_len, mul_mat_vec_nc_f16_f32_data, "main", mul_mat_vec_num_bindings, sizeof(vk_mat_vec_nc_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_norm_f32, "norm_f32", norm_f32_len, norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_group_norm_f32, "group_norm_f32", group_norm_f32_len, group_norm_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_f32, "rms_norm_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_f32, "rms_norm_mul_f32", rms_norm_f32_len, rms_norm_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_partials_f32, "rms_norm_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 0}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_partials_f32, "rms_norm_mul_partials_f32", rms_norm_partials_f32_len, rms_norm_partials_f32_data, "main", 4, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {0, 1}, 1, true);

    if (sizeof(vk_op_rms_norm_mul_rope_push_constants) <= device->properties.limits.maxPushConstantsSize) {
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f32, "rms_norm_mul_rope_f32_f32", rms_norm_mul_rope_f32_f32_len, rms_norm_mul_rope_f32_f32_data, "main", 7, sizeof(vk_op_rms_norm_mul_rope_push_constants), {1, 1, 1}, {0, 1}, 1, true);
        ggml_vk_create_pipeline(device, device->pipeline_rms_norm_mul_rope_f32_f16, "rms_norm_mul_rope_f32_f16", rms_norm_mul_rope_f32_f16_len, rms_norm_mul_rope_f32_f16_data, "main", 7, sizeof(vk_op_rms_norm_mul_rope_push_constants), {1, 1, 1}, {0, 1}, 1, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_rms_norm_back_f32, "rms_norm_back_f32", rms_norm_back_f32_len, rms_norm_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_l2_norm_f32, "l2_norm_f32", l2_norm_f32_len, l2_norm_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f32, "cpy_f32_f32", cpy_f32_f32_len, cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_f16, "cpy_f32_f16", cpy_f32_f16_len, cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f16, "cpy_f16_f16", cpy_f16_f16_len, cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f16_f32, "cpy_f16_f32", cpy_f16_f32_len, cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_bf16,"cpy_f32_bf16",cpy_f32_bf16_len,cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_bf16_f32,"cpy_bf16_f32",cpy_bf16_f32_len,cpy_bf16_f32_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_i32_f32, "cpy_i32_f32", cpy_i32_f32_len, cpy_i32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_i32, "cpy_f32_i32", cpy_f32_i32_len, cpy_f32_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f32, "contig_cpy_f32_f32", contig_cpy_f32_f32_len, contig_cpy_f32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_f16, "contig_cpy_f32_f16", contig_cpy_f32_f16_len, contig_cpy_f32_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f16, "contig_cpy_f16_f16", contig_cpy_f16_f16_len, contig_cpy_f16_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f16_f32, "contig_cpy_f16_f32", contig_cpy_f16_f32_len, contig_cpy_f16_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_bf16,"contig_cpy_f32_bf16",contig_cpy_f32_bf16_len,contig_cpy_f32_bf16_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_bf16_f32,"contig_cpy_bf16_f32",contig_cpy_bf16_f32_len,contig_cpy_bf16_f32_data,"main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_i32_f32, "contig_cpy_i32_f32", contig_cpy_i32_f32_len, contig_cpy_i32_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_contig_cpy_f32_i32, "contig_cpy_f32_i32", contig_cpy_f32_i32_len, contig_cpy_f32_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_32, "cpy_transpose_32", cpy_transpose_32_len, cpy_transpose_32_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_transpose_16, "cpy_transpose_16", cpy_transpose_16_len, cpy_transpose_16_data, "main", 2, sizeof(vk_op_unary_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q1_0], "cpy_f32_q1_0", cpy_f32_q1_0_len, cpy_f32_q1_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_0], "cpy_f32_q4_0", cpy_f32_q4_0_len, cpy_f32_q4_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q4_1], "cpy_f32_q4_1", cpy_f32_q4_1_len, cpy_f32_q4_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_0], "cpy_f32_q5_0", cpy_f32_q5_0_len, cpy_f32_q5_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q5_1], "cpy_f32_q5_1", cpy_f32_q5_1_len, cpy_f32_q5_1_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_Q8_0], "cpy_f32_q8_0", cpy_f32_q8_0_len, cpy_f32_q8_0_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_f32_quant[GGML_TYPE_IQ4_NL], "cpy_f32_iq4_nl", cpy_f32_iq4_nl_len, cpy_f32_iq4_nl_data, "main", 2, sizeof(vk_op_unary_push_constants), {32, 1, 1}, {}, 1);

#define SET_ROWS(itype) \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_F32],  "set_rows_f32" #itype,  set_rows_f32 ## itype ## _len,  set_rows_f32 ## itype ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_F16],  "set_rows_f16" #itype,  set_rows_f16 ## itype ## _len,  set_rows_f16 ## itype ## _data,  "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_BF16], "set_rows_bf16" #itype, set_rows_bf16 ## itype ## _len, set_rows_bf16 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q1_0], "set_rows_q1_0" #itype, set_rows_q1_0 ## itype ## _len, set_rows_q1_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q4_0], "set_rows_q4_0" #itype, set_rows_q4_0 ## itype ## _len, set_rows_q4_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q4_1], "set_rows_q4_1" #itype, set_rows_q4_1 ## itype ## _len, set_rows_q4_1 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q5_0], "set_rows_q5_0" #itype, set_rows_q5_0 ## itype ## _len, set_rows_q5_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q5_1], "set_rows_q5_1" #itype, set_rows_q5_1 ## itype ## _len, set_rows_q5_1 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_Q8_0], "set_rows_q8_0" #itype, set_rows_q8_0 ## itype ## _len, set_rows_q8_0 ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true); \
        ggml_vk_create_pipeline(device, device->pipeline_set_rows ## itype [GGML_TYPE_IQ4_NL], "set_rows_iq4_nl" #itype, set_rows_iq4_nl ## itype ## _len, set_rows_iq4_nl ## itype ## _data, "main", 3, sizeof(vk_op_binary_push_constants), {1, 1, 1}, {1}, 1, true);

    SET_ROWS(_i32)
    SET_ROWS(_i64)
#undef SET_ROWS


    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q1_0], "cpy_q1_0_f32", cpy_q1_0_f32_len, cpy_q1_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q1_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_0], "cpy_q4_0_f32", cpy_q4_0_f32_len, cpy_q4_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q4_1], "cpy_q4_1_f32", cpy_q4_1_f32_len, cpy_q4_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q4_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_0], "cpy_q5_0_f32", cpy_q5_0_f32_len, cpy_q5_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q5_1], "cpy_q5_1_f32", cpy_q5_1_f32_len, cpy_q5_1_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q5_1), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_Q8_0], "cpy_q8_0_f32", cpy_q8_0_f32_len, cpy_q8_0_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_Q8_0), 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cpy_quant_f32[GGML_TYPE_IQ4_NL], "cpy_iq4_nl_f32", cpy_iq4_nl_f32_len, cpy_iq4_nl_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {(uint32_t)ggml_blck_size(GGML_TYPE_IQ4_NL), 1, 1}, {}, 1);

    auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        return s;
    };

#define CREATE_BINARY(name, namemod, spec, bindings) \
    for (int s0 : {0,1}) for (int s1 : {0,1}) for (int d : {0,1}) \
        ggml_vk_create_pipeline2(device, device->pipeline_ ## name ## namemod[s0][s1][d], \
                                #name + get_suffix(s0, s1, d) + #namemod, name ## _len[s0][s1][d], name ## _data[s0][s1][d], \
                                "main", (bindings), sizeof(vk_op_binary_push_constants), {512, 1, 1}, spec, 1);

    CREATE_BINARY(add, , {0}, 4)
    CREATE_BINARY(add, _norepeat, {1}, 4)
    CREATE_BINARY(sub, , {0}, 3)
    CREATE_BINARY(sub, _norepeat, {1}, 3)
    CREATE_BINARY(mul, , {0}, 3)
    CREATE_BINARY(mul, _norepeat, {1}, 3)
    CREATE_BINARY(div, , {0}, 3)
    CREATE_BINARY(div, _norepeat, {1}, 3)
    CREATE_BINARY(add_rms, , {0}, 4)
    CREATE_BINARY(add_rms, _norepeat, {1}, 4)
#undef CREATE_BINARY

    if (device->multi_add) {
        for (uint32_t i = 0; i < MAX_FUSED_ADDS; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add[i],     "multi_add_f32_"     + std::to_string(i+1), multi_add_f32_len,     multi_add_f32_data,     "main", MAX_PARAMETER_COUNT, sizeof(vk_op_multi_add_push_constants), {512, 1, 1}, {i+2}, 1);
            ggml_vk_create_pipeline2(device, device->pipeline_multi_add_rms[i], "multi_add_rms_f32_" + std::to_string(i+1), multi_add_rms_f32_len, multi_add_rms_f32_data, "main", MAX_PARAMETER_COUNT, sizeof(vk_op_multi_add_push_constants), {512, 1, 1}, {i+2}, 1);
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_add_id_f32, "add_id_f32", add_id_f32_len, add_id_f32_data, "main", 4, sizeof(vk_op_add_id_push_constants), {1, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_acc_f32, "acc_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_set_f32, "set_f32", acc_f32_len, acc_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {0, 0}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_concat_i8, "concat_i8", concat_i8_len, concat_i8_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i16, "concat_i16", concat_i16_len, concat_i16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i32, "concat_i32", concat_i32_len, concat_i32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_concat_i64, "concat_i64", concat_i64_len, concat_i64_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_upscale_nearest_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_NEAREST}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bicubic_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BICUBIC}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_upscale_bilinear_antialias_f32, "upscale_f32", upscale_f32_len, upscale_f32_data, "main", 2, sizeof(vk_op_upscale_push_constants), {512, 1, 1}, {GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_scale_f32, "scale_f32", scale_f32_len, scale_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sqr_f32, "sqr_f32", sqr_f32_len, sqr_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sqrt_f32, "sqrt_f32", sqrt_f32_len, sqrt_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_sin_f32, "sin_f32", sin_f32_len, sin_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_cos_f32, "cos_f32", cos_f32_len, cos_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_log[0], "log_f32", log_f32_len, log_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_log[1], "log_f16", log_f16_len, log_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_tri[0], "tri_f32", tri_f32_len, tri_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_tri[1], "tri_f16", tri_f16_len, tri_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag[0], "diag_f32", diag_f32_len, diag_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_diag[1], "diag_f16", diag_f16_len, diag_f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_clamp_f32, "clamp_f32", clamp_f32_len, clamp_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pad_f32, "pad_f32", pad_f32_len, pad_f32_data, "main", 2, sizeof(vk_op_pad_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_roll_f32, "roll_f32", roll_f32_len, roll_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i32, "repeat_i32", repeat_i32_len, repeat_i32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_repeat_back_f32, "repeat_back_f32", repeat_back_f32_len, repeat_back_f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_repeat_i16, "repeat_i16", repeat_i16_len, repeat_i16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_UNARY(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 2, sizeof(vk_op_unary_push_constants), {512, 1, 1}, {}, 1);

    CREATE_UNARY(elu)
    CREATE_UNARY(gelu)
    CREATE_UNARY(gelu_erf)
    CREATE_UNARY(gelu_quick)
    CREATE_UNARY(silu)
    CREATE_UNARY(relu)
    CREATE_UNARY(xielu)
    CREATE_UNARY(neg)
    CREATE_UNARY(tanh)
    CREATE_UNARY(sigmoid)
    CREATE_UNARY(hardsigmoid)
    CREATE_UNARY(hardswish)
    CREATE_UNARY(abs)
    CREATE_UNARY(softplus)
    CREATE_UNARY(step)
    CREATE_UNARY(round)
    CREATE_UNARY(ceil)
    CREATE_UNARY(floor)
    CREATE_UNARY(trunc)
    CREATE_UNARY(sgn)
    CREATE_UNARY(exp)
    CREATE_UNARY(expm1)
#undef CREATE_UNARY

    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f16, "add1_f16_f16", add1_f16_f16_len, add1_f16_f16_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f16_f32, "add1_f16_f32", add1_f16_f32_len, add1_f16_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_add1_f32_f32, "add1_f32_f32", add1_f32_f32_len, add1_f32_f32_data, "main", 3, sizeof(vk_op_binary_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_arange_f32, "arange_f32", arange_f32_len, arange_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_fill_f32, "fill_f32", fill_f32_len, fill_f32_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_fill_f16, "fill_f16", fill_f16_len, fill_f16_data, "main", 1, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

#define CREATE_GLU(name)  \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [0], #name "_f32", name ## _f32_len, name ## _f32_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_ ## name [1], #name "_f16", name ## _f16_len, name ## _f16_data, "main", 3, sizeof(vk_op_glu_push_constants), {512, 1, 1}, {}, 1, true);

    CREATE_GLU(geglu)
    CREATE_GLU(reglu)
    CREATE_GLU(swiglu)
    CREATE_GLU(swiglu_oai)
    CREATE_GLU(geglu_erf)
    CREATE_GLU(geglu_quick)
#undef CREATE_GLU

    ggml_vk_create_pipeline(device, device->pipeline_leaky_relu_f32, "leaky_relu_f32", leaky_relu_f32_len, leaky_relu_f32_data, "main", 2, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_silu_back_f32, "silu_back_f32", silu_back_f32_len, silu_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_diag_mask_inf_f32, "diag_mask_inf_f32", diag_mask_inf_f32_len, diag_mask_inf_f32_data, "main", 2, sizeof(vk_op_diag_mask_push_constants), {1, 512, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32, "soft_max_f32", soft_max_f32_len, soft_max_f32_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_wg512, "soft_max_f32_wg512", soft_max_f32_len, soft_max_f32_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16, "soft_max_f32_f16", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_f32_f16_wg512, "soft_max_f32_f16_wg512", soft_max_f32_f16_len, soft_max_f32_f16_data, "main", 4, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 512 }, 1);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_back_f32, "soft_max_back_f32", soft_max_back_f32_len, soft_max_back_f32_data, "main", 3, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32,     "soft_max_large1_f32",     soft_max_large1_f32_len,     soft_max_large1_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32,     "soft_max_large2_f32",     soft_max_large2_f32_len,     soft_max_large2_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32,     "soft_max_large3_f32",     soft_max_large3_f32_len,     soft_max_large3_f32_data,     "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large1_f32_f16, "soft_max_large1_f32_f16", soft_max_large1_f32_f16_len, soft_max_large1_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large2_f32_f16, "soft_max_large2_f32_f16", soft_max_large2_f32_f16_len, soft_max_large2_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_soft_max_large3_f32_f16, "soft_max_large3_f32_f16", soft_max_large3_f32_f16_len, soft_max_large3_f32_f16_data, "main", 6, sizeof(vk_op_soft_max_push_constants), {1, 1, 1}, { 128, 4 }, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32, "rope_norm_f32", rope_norm_f32_len, rope_norm_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32, "rope_neox_f32", rope_neox_f32_len, rope_neox_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32, "rope_multi_f32", rope_multi_f32_len, rope_multi_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f32, "rope_vision_f32", rope_vision_f32_len, rope_vision_f32_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f16, "rope_norm_f16", rope_norm_f16_len, rope_norm_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f16, "rope_neox_f16", rope_neox_f16_len, rope_neox_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f16, "rope_multi_f16", rope_multi_f16_len, rope_multi_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_vision_f16, "rope_vision_f16", rope_vision_f16_len, rope_vision_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rope_norm_f32_f16, "rope_norm_f32_f16", rope_norm_f32_f16_len, rope_norm_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_neox_f32_f16, "rope_neox_f32_f16", rope_neox_f32_f16_len, rope_neox_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_rope_multi_f32_f16, "rope_multi_f32_f16", rope_multi_f32_f16_len, rope_multi_f32_f16_data, "main", 5, sizeof(vk_op_rope_push_constants), {1, 512, 1}, {}, 1);

    for (uint32_t i = 0; i < num_argsort_pipelines; ++i) {
        uint32_t BLOCK_SIZE = 1u << std::min(i, device->max_workgroup_size_log2);
        if (i <= device->max_workgroup_size_log2 &&
            2 * sizeof(int) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
            const uint32_t NCOLS_PADDED_LOG2 = i;
            ggml_vk_create_pipeline2(device, device->pipeline_argsort_f32[i], "argsort_f32_"+std::to_string(i), argsort_f32_len, argsort_f32_data, "main", 3, sizeof(vk_op_argsort_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, true);
        }
        const uint32_t WG_UNROLL_FACTOR = BLOCK_SIZE > 1 ? 2 : 1;
        BLOCK_SIZE /= WG_UNROLL_FACTOR;
        ggml_vk_create_pipeline2(device, device->pipeline_argsort_large_f32[i], "argsort_large_f32_"+std::to_string(i), argsort_large_f32_len, argsort_large_f32_data, "main", 3, sizeof(vk_op_argsort_push_constants), {BLOCK_SIZE * WG_UNROLL_FACTOR, 1, 1}, {BLOCK_SIZE, WG_UNROLL_FACTOR}, 1, true);
    }

    for (uint32_t i = 0; i < num_topk_pipelines; ++i) {
        const uint32_t BLOCK_SIZE = 1u << i;
        const uint32_t NCOLS_PADDED_LOG2 = i;
        if (i <= device->max_workgroup_size_log2) {
            uint32_t nary_shmem = 2 * sizeof(int) * BLOCK_SIZE +
                                  sizeof(int) * device->subgroup_size +
                                  2 * sizeof(int) +
                                  2 * (BLOCK_SIZE / device->subgroup_size) * sizeof(int);
            if (device->subgroup_arithmetic && device->subgroup_require_full_support && device->subgroup_shuffle && device->subgroup_ballot &&
                nary_shmem <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_nary_search_f32_len, topk_nary_search_f32_data, "main", 2, sizeof(vk_op_topk_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, device->subgroup_size, device->subgroup_size_log2}, 1, true, true, device->subgroup_size);
            } else if (2 * sizeof(int) * BLOCK_SIZE <= device->properties.limits.maxComputeSharedMemorySize) {
                ggml_vk_create_pipeline2(device, device->pipeline_topk_f32[i], "topk_f32_"+std::to_string(i), topk_argsort_f32_len, topk_argsort_f32_data, "main", 2, sizeof(vk_op_topk_push_constants), {BLOCK_SIZE, 1, 1}, {BLOCK_SIZE, NCOLS_PADDED_LOG2}, 1, true);
            }
        }
    }

    ggml_vk_create_pipeline(device, device->pipeline_argmax_f32, "argmax_f32", argmax_f32_len, argmax_f32_data, "main", 2, sizeof(vk_op_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_sum_rows_f32, "sum_rows_f32", sum_rows_f32_len, sum_rows_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { device->subgroup_size }, 1);
    // Intel Arc B390 was observed segfaulting with this shader.
    if (device->subgroup_basic && device->subgroup_shuffle && device->vendor_id != VK_VENDOR_ID_INTEL) {
        int idx = 0;
        for (uint32_t n : {64, 128, 256, 512}) {
            if (device->subgroup_size <= n) {
                ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n }, 1, true, true, device->subgroup_size);
            }
            ++idx;
        }
    } else if (device->driver_id != vk::DriverId::eIntelProprietaryWindows) {
        // Disabled on Intel Windows due to a driver bug: https://github.com/ggml-org/llama.cpp/pull/23964#issuecomment-4598226147
        int idx = 0;
        for (uint32_t n : {64, 128, 256, 512}) {
            const uint32_t block_size = std::min(device->subgroup_size, n);
            ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n }, 1);
            ++idx;
        }
    }

    const uint32_t cumsum_elem_per_thread = (device->vendor_id == VK_VENDOR_ID_AMD || device->vendor_id == VK_VENDOR_ID_INTEL) ? 2 : 4;
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_f32,       "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { 256, device->subgroup_size, cumsum_elem_per_thread }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_small_f32, "cumsum_f32", cumsum_f32_len, cumsum_f32_data, "main", 2, sizeof(vk_op_sum_rows_push_constants), {1, 1, 1}, { 128, device->subgroup_size, 1 }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass1_f32, "cumsum_multipass1_f32", cumsum_multipass1_f32_len, cumsum_multipass1_f32_data, "main", 3, sizeof(vk_op_sum_rows_push_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, true, true, device->subgroup_size);
    ggml_vk_create_pipeline(device, device->pipeline_cumsum_multipass2_f32, "cumsum_multipass2_f32", cumsum_multipass2_f32_len, cumsum_multipass2_f32_data, "main", 3, sizeof(vk_op_sum_rows_push_constants), {256, 1, 1}, { 256, device->subgroup_size }, 1, true, true, device->subgroup_size);

    ggml_vk_create_pipeline(device, device->pipeline_count_equal_i32, "count_equal_i32", count_equal_i32_len, count_equal_i32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, { device->subgroup_size }, 1);

    ggml_vk_create_pipeline(device, device->pipeline_count_experts, "count_experts", count_experts_len, count_experts_data, "main", 2, sizeof(vk_op_count_experts_push_constants), {1, 1, 1}, {}, 1, true);

    for (auto &s : device->pipeline_solve_tri_f32) {
        const vk_solve_tri_pipeline_state &state = s.first;

        // Max number of rows to load at a time, limited by shared memory
        const uint32_t batch_N = device->properties.limits.maxComputeSharedMemorySize / ((state.N + state.K) * sizeof(float));
        // Need at least K invocations, and prefer a minimum of 128 to spread out loading shared memory
        const uint32_t block_size = std::max(128u, 1u << (uint32_t)ceilf(log2f(float(state.K))));

        ggml_vk_create_pipeline(
            device, s.second, "solve_tri_f32",
            solve_tri_f32_len, solve_tri_f32_data, "main", 3,
            sizeof(vk_op_binary_push_constants), {1, 1, 1}, { 0, state.N, state.K, batch_N, block_size }, 1, true);
    }

#define IM2COL(bda) \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32, "im2col_f32", im2col_f32 ## bda ## _len, im2col_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32, "im2col_3d_f32", im2col_3d_f32 ## bda ## _len, im2col_3d_f32 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);      \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_f32_f16, "im2col_f32_f16", im2col_f32_f16 ## bda ## _len, im2col_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_push_constants), {512, 1, 1}, { device->subgroup_size }, 1, true);   \
    ggml_vk_create_pipeline(device, device->pipeline_im2col_3d_f32_f16, "im2col_3d_f32_f16", im2col_3d_f32_f16 ## bda ## _len, im2col_3d_f32_f16 ## bda ## _data, "main", 2, sizeof(vk_op_im2col_3d_push_constants), {512, 1, 1}, { 512 }, 1, true);
    if (device->shader_int64 && device->buffer_device_address) {
        IM2COL(_bda)
    } else {
        IM2COL()
    }

    ggml_vk_create_pipeline(device, device->pipeline_timestep_embedding_f32, "timestep_embedding_f32", timestep_embedding_f32_len, timestep_embedding_f32_data, "main", 2, sizeof(vk_op_timestep_embedding_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_conv_transpose_1d_f32, "conv_transpose_1d_f32", conv_transpose_1d_f32_len, conv_transpose_1d_f32_data, "main", 3, sizeof(vk_op_conv_transpose_1d_push_constants), {1, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f32,  "col2im_1d_f32",  col2im_1d_f32_len,  col2im_1d_f32_data,  "main", 2, sizeof(vk_op_col2im_1d_push_constants), {256, 1, 1}, {}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_f16,  "col2im_1d_f16",  col2im_1d_f16_len,  col2im_1d_f16_data,  "main", 2, sizeof(vk_op_col2im_1d_push_constants), {256, 1, 1}, {}, 1, true);
    ggml_vk_create_pipeline(device, device->pipeline_col2im_1d_bf16, "col2im_1d_bf16", col2im_1d_bf16_len, col2im_1d_bf16_data, "main", 2, sizeof(vk_op_col2im_1d_push_constants), {256, 1, 1}, {}, 1, true);

    ggml_vk_create_pipeline(device, device->pipeline_snake_f32,  "snake_f32",  snake_f32_len,  snake_f32_data,  "main", 4, sizeof(vk_op_snake_push_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_f16,  "snake_f16",  snake_f16_len,  snake_f16_data,  "main", 4, sizeof(vk_op_snake_push_constants), {256, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_snake_bf16, "snake_bf16", snake_bf16_len, snake_bf16_data, "main", 4, sizeof(vk_op_snake_push_constants), {256, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_pool2d_f32, "pool2d_f32", pool2d_f32_len, pool2d_f32_data, "main", 2, sizeof(vk_op_pool2d_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv6_f32, "rwkv_wkv6_f32", rwkv_wkv6_f32_len, rwkv_wkv6_f32_data, "main", 7, sizeof(vk_op_rwkv_wkv6_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_rwkv_wkv7_f32, "rwkv_wkv7_f32", rwkv_wkv7_f32_len, rwkv_wkv7_f32_data, "main", 8, sizeof(vk_op_rwkv_wkv7_push_constants), {1, 1, 1}, {device->subgroup_size}, 1);

    {
        const uint32_t gdn_sizes[] = {16, 32, 64, 128};
        const char * gdn_names[][2] = {
            {"gated_delta_net_f32_d16",     "gated_delta_net_f32_d16_kda"},
            {"gated_delta_net_f32_d32",     "gated_delta_net_f32_d32_kda"},
            {"gated_delta_net_f32_d64",     "gated_delta_net_f32_d64_kda"},
            {"gated_delta_net_f32_d128",    "gated_delta_net_f32_d128_kda"},
        };
        for (uint32_t si = 0; si < 4; si++) {
            const uint32_t S_V = gdn_sizes[si];
            GGML_ASSERT(is_pow2(S_V));

            uint32_t lanes_per_column;
            if (S_V >= 128u && device->subgroup_clustered) {
                lanes_per_column = 8u;
            } else {
                // Use largest power-of-two that divides both S_V and subgroup_size so that
                // (1) S_V % lanes_per_column == 0 and (2) S_V % (subgroup_size / lanes_per_column) == 0.
                // This means we don't need extra bounds checking logic in the shader.
                lanes_per_column = std::min(S_V, device->subgroup_size);
            }

            // gated_delta_net.comp relies on S_V % COLS_PER_WG == 0 and
            // S_V % LANES_PER_COLUMN == 0 to avoid bounds checks.
            while (lanes_per_column > 1u) {
                const bool valid_lanes = (device->subgroup_size % lanes_per_column) == 0 &&
                                         (S_V % lanes_per_column) == 0;
                const uint32_t cols_per_wg = valid_lanes ? device->subgroup_size / lanes_per_column : 0;
                if (valid_lanes && cols_per_wg > 0 && (S_V % cols_per_wg) == 0) {
                    break;
                }
                lanes_per_column >>= 1u;
            }

            GGML_ASSERT((device->subgroup_size % lanes_per_column) == 0);
            GGML_ASSERT((S_V % lanes_per_column) == 0);
            GGML_ASSERT((S_V % (device->subgroup_size / lanes_per_column)) == 0);

            const bool need_partial_subgroup_reduce = lanes_per_column != 1u && lanes_per_column < device->subgroup_size;
            const bool use_clustered_reduce = device->subgroup_arithmetic && device->subgroup_clustered && need_partial_subgroup_reduce;
            const bool use_subgroup_reduce = device->subgroup_arithmetic && !need_partial_subgroup_reduce;
            const bool use_subgroup_ops = use_clustered_reduce || use_subgroup_reduce;
            size_t gdn_len;
            const void * gdn_data;
            if (use_clustered_reduce) {
                gdn_len = gated_delta_net_f32_len;
                gdn_data = (const void *)gated_delta_net_f32_data;
            } else if (use_subgroup_reduce) {
                gdn_len = gated_delta_net_f32_nocluster_len;
                gdn_data = (const void *)gated_delta_net_f32_nocluster_data;
            } else {
                gdn_len = gated_delta_net_f32_shmem_len;
                gdn_data = (const void *)gated_delta_net_f32_shmem_data;
            }

            const uint32_t cols_per_wg = device->subgroup_size / lanes_per_column;
            const std::array<uint32_t, 3> wg_denoms = {1u, 1u, cols_per_wg};

            for (uint32_t kda = 0; kda < 2; kda++) {
                ggml_vk_create_pipeline(device, device->pipeline_gated_delta_net[si][kda],
                    gdn_names[si][kda], gdn_len, gdn_data, "main", 7, sizeof(vk_op_gated_delta_net_push_constants),
                    wg_denoms, {S_V, kda, device->subgroup_size, lanes_per_column}, 1, true, use_subgroup_ops, device->subgroup_size);
            }
        }
    }

    if (device->subgroup_arithmetic && device->subgroup_require_full_support) {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {128, device->subgroup_size}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_subgroup_f32_len, ssm_scan_subgroup_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {256, device->subgroup_size}, 1, true, true);
    } else {
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d128, "ssm_scan_128_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {128, device->subgroup_size, 16}, 1, true, true);
        ggml_vk_create_pipeline(device, device->pipeline_ssm_scan_f32_d256, "ssm_scan_256_f32", ssm_scan_f32_len, ssm_scan_f32_data, "main", 8, sizeof(vk_op_ssm_scan_push_constants), {1, 1, 1}, {256, device->subgroup_size, 16}, 1, true, true);
    }

    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_f32,           "ssm_conv_f32",           ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16, 0, 0}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_silu_f32,      "ssm_conv_silu_f32",      ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16, 0, 1}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_ssm_conv_bias_silu_f32, "ssm_conv_bias_silu_f32", ssm_conv_f32_len, ssm_conv_f32_data, "main", 4, sizeof(vk_op_ssm_conv_push_constants), {32, 16, 1}, {32, 16, 1, 1}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_adamw_f32, "opt_step_adamw_f32", opt_step_adamw_f32_len, opt_step_adamw_f32_data, "main", 5, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    ggml_vk_create_pipeline(device, device->pipeline_opt_step_sgd_f32, "opt_step_sgd_f32", opt_step_sgd_f32_len, opt_step_sgd_f32_data, "main", 3, sizeof(vk_op_push_constants), {512, 1, 1}, {}, 1);

    // conv2d, conv_transpose_2d
    for (uint32_t s = 0; s < CONV_SHAPE_COUNT; ++s) {
        // smaller WG for the small-tile fallback gives more concurrent WGs per SM
        uint32_t conv2d_WG_SIZE  = (s == CONV_SHAPE_64x32) ? 128 : 256;
        uint32_t use_collectives = 0;  // Enables subgroup ops for preventing the re-calculation of indices.
        uint32_t conv2d_TS_K     = (s == CONV_SHAPE_64x32) ? 4 : 8;
        uint32_t conv2d_SHMEM_PAD = 4;
        vk_conv_block_size conv2d_BS = vk_conv_block_sizes[s];
        bool conv2d_UNROLL = true;

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            conv2d_SHMEM_PAD = 8; // 8 float16_t
        }
#endif

        if (device->vendor_id == VK_VENDOR_ID_INTEL) {
            conv2d_SHMEM_PAD = 0;
            conv2d_UNROLL = false;
        } else if (device->vendor_id == VK_VENDOR_ID_AMD) {
            conv2d_SHMEM_PAD = device->architecture == vk_device_architecture::AMD_GCN ? 1 : 4;
            if (s == CONV_SHAPE_128x128 && device->architecture != vk_device_architecture::AMD_GCN) {
                conv2d_UNROLL = false;
            }
        }

        // Use collectives on pre-Turing NVIDIA GPUs and GCN AMD cards, which had slower integer math.
        bool allow_collectives_nv = device->vendor_id != VK_VENDOR_ID_NVIDIA ||
                                    device->architecture == vk_device_architecture::NVIDIA_PRE_TURING;
        bool allow_collectives_amd = device->vendor_id != VK_VENDOR_ID_AMD ||
                                     device->architecture == vk_device_architecture::AMD_GCN;

        if (device->subgroup_shuffle &&
            device->vendor_id != VK_VENDOR_ID_INTEL &&   // Do not enable collectives on Intel, see PR 14316.
            allow_collectives_nv &&
            allow_collectives_amd) {
            use_collectives = 1;
            conv2d_BS.CRS   = std::min(
                device->subgroup_size,
                conv2d_BS.CRS);  // CRS block size should be capped at subgroup size for correctness when shuffle is used.
        }

        // cm1 is used only when cm2 is unavailable; capped at 64x128 (due to shared memory size).
        // Requires 16x16x16 f16-acc since that's the fragment shape hard-coded in the shader.
        // Subgroup size must be 32 or 64 (to keep WG_SIZE sane) and we need
        // subgroup_size_control to force the driver to actually use it.
        bool conv2d_use_cm1 = false;
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        conv2d_use_cm1 = !device->coopmat2 &&
                         device->coopmat_support && device->coopmat_support_16x16x16_f16acc &&
                         device->subgroup_size_control &&
                         (device->subgroup_size == 32 || device->subgroup_size == 64) &&
                         s != CONV_SHAPE_128x128;
#endif

        const uint32_t conv2d_cm1_shmem_pad = 8;

        auto shmem_req = [&](uint32_t pad, bool csh_store, bool fp16_shmem) {
            const uint32_t elem_size = fp16_shmem ? (uint32_t)sizeof(uint16_t) : (uint32_t)sizeof(float);
            const uint32_t csh_elems = csh_store ? conv2d_BS.K * conv2d_BS.NPQ : 0u;
            return (conv2d_BS.K * (conv2d_BS.CRS + pad) + conv2d_BS.CRS * (conv2d_BS.NPQ + pad) + csh_elems) * elem_size;
        };

        // coopmat1 needs to store the output through shared memory, so check up front
        // whether it'll fit and disable it before applying coopmat1 parameters.
        if (conv2d_use_cm1 && device->properties.limits.maxComputeSharedMemorySize < shmem_req(conv2d_cm1_shmem_pad, true, true)) {
            conv2d_use_cm1 = false;
        }

        uint32_t conv2d_WM = 16, conv2d_WN = 16;  // cm1 subgroup tile, ignored otherwise
        if (conv2d_use_cm1) {
            conv2d_SHMEM_PAD = conv2d_cm1_shmem_pad;
            // 16x16x16 fragments; pick WM/WN to keep WG_SIZE at 256
            // (i.e. 8 subgroups for sg=32, 4 subgroups for sg=64).
            const bool sg64 = (device->subgroup_size == 64);
            switch (s) {
                case CONV_SHAPE_64x32:   conv2d_WM = sg64 ? 32 : 16; conv2d_WN = 16; break;
                case CONV_SHAPE_64x128:  conv2d_WM = 32; conv2d_WN = sg64 ? 64 : 32; break;
                case CONV_SHAPE_32x256:  conv2d_WM = sg64 ? 16 : 32; conv2d_WN = sg64 ? 128 : 32; break;
                default: break;
            }
            const uint32_t warps_M = conv2d_BS.K / conv2d_WM;
            const uint32_t warps_N = conv2d_BS.NPQ / conv2d_WN;
            conv2d_WG_SIZE         = warps_M * warps_N * device->subgroup_size;
        }

        // stage cm2 accumulator through shmem for coalesced global stores;
        // skipped on 128x128 where the extra Csh footprint hurts occupancy.
        // cm1 always uses the staged path.
        uint32_t conv2d_csh_store = (device->coopmat2 && s != CONV_SHAPE_128x128) ? 1u : 0u;
        if (conv2d_use_cm1) {
            conv2d_csh_store = 1;
        }

        // shmem is fp16 on cm2/cm1 (matches Csh), fp32 on scalar
        const bool conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1;

        // shrink CRS if the non-cm1 config still doesn't fit
        if (device->properties.limits.maxComputeSharedMemorySize < shmem_req(conv2d_SHMEM_PAD, conv2d_csh_store, conv2d_use_fp16_shmem)) {
            GGML_ASSERT(!conv2d_use_cm1);
            conv2d_BS.CRS = 8;
            if (use_collectives) {
                conv2d_BS.CRS = std::min(device->subgroup_size, conv2d_BS.CRS);
            }
            conv2d_csh_store = 0;
        }

        std::array<uint32_t, 3> wg_denoms = { conv2d_BS.K, 1, 1 };
        std::vector<uint32_t> spec_constants = { conv2d_WG_SIZE, conv2d_BS.K, conv2d_BS.CRS, conv2d_BS.NPQ, conv2d_TS_K, use_collectives, conv2d_SHMEM_PAD };

        // cm1 needs a fixed subgroup width to match the WG_SIZE we computed
        const uint32_t conv2d_required_subgroup_size = conv2d_use_cm1 ? device->subgroup_size : 0;

#define CREATE_CONV(name, type_suffix, spv_suffix) \
        for (auto &c : device->pipeline_##name##type_suffix[s]) { \
            const vk_conv2d_pipeline_state &state = c.first;  \
            std::vector<uint32_t> spec_constants_cpy = spec_constants; \
            spec_constants_cpy.push_back(state.s0); \
            spec_constants_cpy.push_back(state.s1); \
            spec_constants_cpy.push_back(state.p0); \
            spec_constants_cpy.push_back(state.p1); \
            spec_constants_cpy.push_back(state.d0); \
            spec_constants_cpy.push_back(state.d1); \
            spec_constants_cpy.push_back(state.KW); \
            spec_constants_cpy.push_back(state.KH); \
            spec_constants_cpy.push_back(state.aligned); \
            spec_constants_cpy.push_back(conv2d_csh_store); \
            spec_constants_cpy.push_back(conv2d_WM); \
            spec_constants_cpy.push_back(conv2d_WN); \
            ggml_vk_create_pipeline( \
                device, c.second, #name #type_suffix, \
                name##type_suffix##spv_suffix##_len, name##type_suffix##spv_suffix##_data, "main", 3, \
                sizeof(vk_op_conv2d_push_constants), wg_denoms, spec_constants_cpy, 1, true, use_collectives || conv2d_required_subgroup_size, conv2d_required_subgroup_size);    \
        }
#define CREATE_CONVS(spv_suffix) \
        CREATE_CONV(conv2d, _f32, spv_suffix) \
        CREATE_CONV(conv2d, _f16_f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f32, spv_suffix) \
        CREATE_CONV(conv_transpose_2d, _f16_f32, spv_suffix)
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (device->coopmat2) {
            CREATE_CONVS(_cm2)
        } else
#endif
#if defined(VK_KHR_cooperative_matrix) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        if (conv2d_use_cm1) {
            CREATE_CONVS(_cm1)
        } else
#endif
        if (conv2d_UNROLL) {
            CREATE_CONVS(_unroll)
        } else {
            CREATE_CONVS( )
        }
#undef CREATE_CONV
#undef CREATE_CONVS
    }

    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f32, "conv2d_dw_whcn_f32", conv2d_dw_whcn_f32_len, conv2d_dw_whcn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f32, "conv2d_dw_cwhn_f32", conv2d_dw_cwhn_f32_len, conv2d_dw_cwhn_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_whcn_f16_f32, "conv2d_dw_whcn_f16_f32", conv2d_dw_whcn_f16_f32_len, conv2d_dw_whcn_f16_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);
    ggml_vk_create_pipeline(device, device->pipeline_conv2d_dw_cwhn_f16_f32, "conv2d_dw_cwhn_f16_f32", conv2d_dw_cwhn_f16_f32_len, conv2d_dw_cwhn_f16_f32_data, "main", 3, sizeof(vk_op_conv2d_dw_push_constants), {512, 1, 1}, {}, 1);

    for (uint32_t use_push = 0; use_push < 2; ++use_push) {
        for (uint32_t i = 0; i < num_topk_moe_pipelines; ++i) {
            ggml_vk_create_pipeline2(device, device->pipeline_topk_moe[i][use_push], "topk_moe_f32_"+std::to_string(i), topk_moe_f32_len, topk_moe_f32_data, "main", 4, sizeof(vk_op_topk_moe_push_constants), {1, 1, 1}, {device->subgroup_size, 1u<<i, use_push}, 1, true, true, device->subgroup_size);
        }
    }

    // Drop compile_mutex so other threads can walk while we compile.
    compile_lock.unlock();

    // Compile what we claimed; create_pipeline_func reacquires compile_mutex
    // at the end to flip compile_pending/compiled and notify waiters.
    if (has_claimed_task) {
        auto & task = claimed_task;
        ggml_vk_create_pipeline_func(device, task.pipeline, task.spv_size, task.spv_data,
                                     task.entrypoint, task.parameter_count, task.wg_denoms,
                                     task.specialization_constants, task.disable_robustness,
                                     task.require_full_subgroups, task.required_subgroup_size);
    }

    // Another thread may be compiling the pipeline we need; block on it here.
    if (wait_pipeline) {
        std::unique_lock<std::mutex> wait_lock(device->compile_mutex);
        device->compile_cv.wait(wait_lock, [&] {
            return wait_pipeline->compiled.load();
        });
    }
}

bool ggml_vk_flash_attn_scalar_shmem_support(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type, ggml_type v_type) {
    GGML_UNUSED(f32acc);
    GGML_UNUSED(v_type);
    // Needs to be kept up to date on shader changes
    const uint32_t wg_size = params.workgroup_size;
    const uint32_t Br = params.block_rows;
    const uint32_t Bc = params.block_cols;

    // BF16 uses the fp32 shader (FLOAT_TYPE=float)
    const uint32_t float_type_size = (device->fp16 && k_type != GGML_TYPE_BF16) ? sizeof(ggml_fp16_t) : sizeof(float);

    const bool mmq = ggml_vk_fa_scalar_uses_mmq(device, k_type);

    // tmpsh is overestimated slightly
    const uint32_t tmpsh = wg_size * sizeof(float);
    const uint32_t tmpshv4 = wg_size * 4 * float_type_size;

    const uint32_t masksh = Bc * (Br + 1) * float_type_size;

    uint32_t Qf, kvsh, kblocksh_size;
    if (mmq) {
        // block_b_cache: int32_t qs[8] + FLOAT_TYPEV2 ds
        const uint32_t block_b_size = 8 * sizeof(int32_t) + 2 * float_type_size;
        Qf = Br * (hsk / 32) * block_b_size;

        // kvsh uses D = HSV (K goes through kblocksh instead)
        kvsh = params.shmem_staging ? Bc * (hsv / 4 + 1) * 4 * float_type_size : 4 * float_type_size;

        // The mixed MMQ shader uses a superset block_a_cache that fits every
        // FA-supported quant: int32_t qs[8] + uint32_t qh + FLOAT_TYPEV2 dm.
        // Single-scale types leave dm.y unused; non-Q5_* leave qh unused.
        const uint32_t block_a_size = 8 * sizeof(int32_t) + sizeof(uint32_t) + 2 * float_type_size;
        kblocksh_size = params.shmem_staging ? Bc * (hsk / 32) * block_a_size : block_a_size;
    } else {
        Qf = Br * (hsk / 4 + 1) * 4 * float_type_size;

        const uint32_t D = std::max(hsk, hsv);
        kvsh = params.shmem_staging ? Bc * (D / 4 + 1) * 4 * float_type_size : 4 * float_type_size;

        kblocksh_size = 0;
    }

    const uint32_t total_size = tmpsh + tmpshv4 + masksh + Qf + kvsh + kblocksh_size;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_scalar_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", mmq=" << mmq << ", total_size=" << total_size << ", supported=" << supported);

    return supported;
}

bool ggml_vk_flash_attn_coopmat_shmem_support(const vk_device& device, const vk_fa_tuning_params& params, uint32_t hsk, uint32_t hsv, bool f32acc, ggml_type k_type) {
    // Needs to be kept up to date on shader changes
    const uint32_t Br = params.block_rows;
    const uint32_t Bc = params.block_cols;

    const uint32_t MatBr = 16, MatBc = 16;

    const uint32_t row_split = Bc / MatBc;

    const uint32_t hsk_pad = ROUNDUP_POW2(hsk, 16);
    const uint32_t hsv_pad = ROUNDUP_POW2(hsv, 16);

    const uint32_t acctype = f32acc ? 4 : 2;
    const uint32_t f16vec4 = 8;

    const uint32_t tmpsh = (Bc / MatBc) * sizeof(float);

    const uint32_t qstride = hsk_pad / 4 + 2;
    const uint32_t Qf = Br * qstride * f16vec4;

    const uint32_t psh_stride = Br / 4 + 2;
    const uint32_t Psh = Bc * psh_stride * f16vec4;

    const uint32_t sfshstride = (hsk <= 128) ? (Br + 8) : Br;
    const uint32_t sfsh = Bc * sfshstride * acctype;

    const uint32_t kvshstride = (params.shmem_staging ? std::max(hsk_pad, hsv_pad) : MatBr) / 4 + 2;
    const uint32_t vsh_stride = MatBc / 4 * row_split;
    const uint32_t ksh = ((kvshstride >= vsh_stride) ? (Bc * kvshstride) : (Bc * vsh_stride)) * f16vec4;

    // BF16 PVMat accumulator is f32 (no bf16 accumulator support), so pvsh is vec4 (16 bytes)
    const uint32_t pvsh_elem_size = (k_type == GGML_TYPE_BF16) ? 16u : f16vec4;
    const uint32_t osh_stride = params.row_split * MatBr / 4;
    const uint32_t pvsh = MatBc * osh_stride * pvsh_elem_size;

    const uint32_t slope = Br * acctype;

    const uint32_t total_size = tmpsh + Qf + Psh + sfsh + ksh + pvsh + slope;
    const bool supported = total_size <= device->properties.limits.maxComputeSharedMemorySize;

    VK_LOG_DEBUG("ggml_vk_flash_attn_coopmat_shmem_support(HSK=" << hsk << ", HSV=" << hsv << ", f32acc=" << f32acc << ", total_size=" << total_size << ", supported=" << supported);

    return supported;
}

