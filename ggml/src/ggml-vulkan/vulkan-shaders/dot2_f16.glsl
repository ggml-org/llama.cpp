#ifdef DOT2_F16
#extension GL_EXT_spirv_intrinsics : require

spirv_instruction(extensions = ["SPV_VALVE_mixed_float_dot_product"],
                  capabilities = [6912], id = 6916)
float v_dot2_f32_f16(f16vec2 a, f16vec2 b, float acc);
#endif
