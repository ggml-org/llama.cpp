void init_iq_shmem(uvec3 wgsize) {
    if (MmTypeA == GGML_TYPE_IQ1_S || MmTypeA == GGML_TYPE_IQ1_M) {
        [[unroll]] for (uint i = 0; i < iq1s_grid_const.length(); i += wgsize.x) {
            uint idx = i + gl_LocalInvocationIndex.x;
            if (iq1s_grid_const.length() % wgsize.x == 0 || idx < iq1s_grid_const.length()) {
                u16vec2 g = unpack16(iq1s_grid_const[idx]);
                iq1s_grid[2*idx+0] = g.x;
                iq1s_grid[2*idx+1] = g.y;
            }
        }
    } else if (MmTypeA == GGML_TYPE_IQ2_XXS) {
        [[unroll]] for (uint i = 0; i < iq2xxs_grid.length(); i += wgsize.x) {
            if (iq2xxs_grid_const.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2xxs_grid_const.length()) {
                iq2xxs_grid[i + gl_LocalInvocationIndex.x] = iq2xxs_grid_const[i + gl_LocalInvocationIndex.x];
            }
        }
    } else if (MmTypeA == GGML_TYPE_IQ2_XS) {
        [[unroll]] for (uint i = 0; i < iq2xs_grid.length(); i += wgsize.x) {
            if (iq2xs_grid_const.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2xs_grid_const.length()) {
                iq2xs_grid[i + gl_LocalInvocationIndex.x] = iq2xs_grid_const[i + gl_LocalInvocationIndex.x];
            }
        }
    } else if (MmTypeA == GGML_TYPE_IQ2_S) {
        [[unroll]] for (uint i = 0; i < iq2s_grid.length(); i += wgsize.x) {
            if (iq2s_grid_const.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2s_grid_const.length()) {
                iq2s_grid[i + gl_LocalInvocationIndex.x] = iq2s_grid_const[i + gl_LocalInvocationIndex.x];
            }
        }
    } else if (MmTypeA == GGML_TYPE_IQ3_XXS) {
        [[unroll]] for (uint i = 0; i < iq3xxs_grid.length(); i += wgsize.x) {
            if (iq3xxs_grid_const.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq3xxs_grid_const.length()) {
                iq3xxs_grid[i + gl_LocalInvocationIndex.x] = iq3xxs_grid_const[i + gl_LocalInvocationIndex.x];
            }
        }
    } else if (MmTypeA == GGML_TYPE_IQ3_S) {
        [[unroll]] for (uint i = 0; i < iq3s_grid.length(); i += wgsize.x) {
            if (iq3s_grid_const.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq3s_grid_const.length()) {
                iq3s_grid[i + gl_LocalInvocationIndex.x] = iq3s_grid_const[i + gl_LocalInvocationIndex.x];
            }
        }
    } else if (MmTypeA == GGML_TYPE_IQ4_XS || MmTypeA == GGML_TYPE_IQ4_NL) {
        for (uint i = gl_LocalInvocationIndex.x; i < kvalues_iq4nl.length(); i += wgsize.x) {
            kvalues_iq4nl[i] = FLOAT_TYPE(kvalues_iq4nl_const[i]);
        }
    }
#if !defined(USE_OCP_FP4)
    else if (MmTypeA == GGML_TYPE_MXFP4 || MmTypeA == GGML_TYPE_NVFP4) {
        for (uint i = gl_LocalInvocationIndex.x; i < kvalues_mxfp4.length(); i += wgsize.x) {
            kvalues_mxfp4[i] = kvalues_mxfp4_const[i];
        }
        if (MmTypeA == GGML_TYPE_NVFP4) {
            for (uint i = gl_LocalInvocationIndex.x; i < 128u; i += wgsize.x) {
                ue4m3_fp32_lut[i] = ue4m3_to_fp32_build(i);
            }
        }
    }
#endif
    barrier();
}
