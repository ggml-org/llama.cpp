# 참고한 1차 자료

1. DeepGrove, `maple-preview-2bit-mlx/maple.py`, 표시된 revision `e8be321`, “Speed up Maple decode with and without the native extension”.
   `https://huggingface.co/deepgrove/maple-preview-2bit-mlx/blob/e8be321/maple.py`
   참고 범위: `_gather_sort`/`_scatter_unsort`, `up_gate_proj`, `qkv_proj`, FP32 router, top-k selection, clamped SwiGLU, reference rounding을 유지하려는 probe/fallback 철학.
2. Khronos, `GLSL_EXT_integer_dot_product`.
   `https://docs.vulkan.org/glslext/latest/glslext/ext/GLSL_EXT_integer_dot_product.html`
   참고 범위: signed `dotPacked4x8EXT`, extension/SPIR-V capability 경로. 원본 SPIR-V 컴파일과 GPU 하드웨어 lower는 이 환경에서 검증하지 못함.
3. 업로드 빌드 산출물의 `build-info.cpp`, `ggml-version.h`, `CMakeCache.txt`, `compile_commands.json` 및 ZIP 중앙 디렉터리.
   원본 ggml 소스는 이 파일들에 포함되어 있지 않았다. 재배포한 것은 작은 provenance 파일과 archive manifest뿐이며 DLL/OBJ/SPIR-V 산출물을 포함하지 않았다.

설계 원칙을 참고한 신규 구현이다. MLX의 Metal source/MLX 2bit packed tensor를 그대로 복사하지 않았으며 그 바이너리 호환성이나 bit-exact를 주장하지 않는다. 참조 프로젝트와 공식적으로 제휴한 패키지가 아니다.
