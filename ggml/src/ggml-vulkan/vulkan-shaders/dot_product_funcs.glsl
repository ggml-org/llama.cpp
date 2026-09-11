#ifdef DOT2_F16
#extension GL_EXT_spirv_intrinsics : require

spirv_instruction(extensions = ["SPV_VALVE_mixed_float_dot_product"],
                  capabilities = [6912], id = 6916)
float v_dot2_f32_f16(f16vec2 a, f16vec2 b, float acc);

ACC_TYPE dot_product(f16vec4 a, f16vec4 b, ACC_TYPE acc) {
    return ACC_TYPE(v_dot2_f32_f16(a.zw, b.zw, v_dot2_f32_f16(a.xy, b.xy, float(acc))));
}

ACC_TYPE dot_product(f16vec2 a, f16vec2 b, ACC_TYPE acc) {
    return ACC_TYPE(v_dot2_f32_f16(a, b, float(acc)));
}

#else

ACC_TYPE dot_product(FLOAT_TYPEV4 a, FLOAT_TYPEV4 b, ACC_TYPE acc) {
    return fma(ACC_TYPE(a.x), ACC_TYPE(b.x), fma(ACC_TYPE(a.y), ACC_TYPE(b.y),
           fma(ACC_TYPE(a.z), ACC_TYPE(b.z), fma(ACC_TYPE(a.w), ACC_TYPE(b.w), acc))));
}

ACC_TYPE dot_product(FLOAT_TYPEV2 a, FLOAT_TYPEV2 b, ACC_TYPE acc) {
    return fma(ACC_TYPE(a.x), ACC_TYPE(b.x), fma(ACC_TYPE(a.y), ACC_TYPE(b.y), acc));
}

#endif
