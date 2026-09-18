# 수행한 검증과 수행하지 못한 검증

## 환경

Linux x86_64 작업 컨테이너. C++ CPU 테스트는 시스템 C++17 컴파일러로 빌드했다.
보조 셰이더 실행은 `llvmpipe (LLVM 19.1.7, 256 bits)`, `OpenGL 4.5 Mesa 25.0.7-2`다.
실제 A750은 없다. `glslc`가 없어 원본 GLSL/SPIR-V 빌드는 수행하지 못했다. Vulkan headers/library 및 device가 없어 `src/vk_replay.cpp`는 이 환경에서 컴파일/실행되지 않았다.

## 결과

| 검사 | 결과 | 의미 |
|---|---|---|
| C++ TQ2 unpack | 1,114,112 lane checks PASS | byte/bitplane mapping 및 signed-byte 표현 |
| C++ packed integer dot | 262,144 checks PASS | packed arithmetic와 scalar reference |
| C++ ASan + UBSan | PASS | 해당 CPU 테스트 실행 중 sanitizer 오류 없음 |
| Python reference | 13 tests PASS | Q8 규약, fusion, routing, permutation, capacity, down/reduce |
| Python 구문 | 모든 제공 `.py` py_compile PASS | syntax 검증 |
| 변환한 GLSL compile/link | 23/23 PASS | Mesa OpenGL target, native dot 치환 |
| F32 N13/H256/F256/E256/S8 | PASS | bucket + persistent + upgate/QKV fusion |
| Q8 N13/H256/F256/E256/S8 | PASS | Q8 + bucket + persistent + fusion |
| F32 N1, no-fusion/no-reuse/full-router | PASS | direct 및 대조 경로, high-ID tie/ascending 옵션 |
| Q8 N13, no-persistent/no-reuse | PASS | static bucket 및 no-reuse 대조 경로 |
| Q8 N184/H256/F256/E256/S8 | PASS | 여러 token tiles/expert |
| Q8 N2048/H256/F256/E256/S8 | PASS | 큰 job queue, 2D quantization dispatch |
| Q8 N1/H2048/F512/E256/S8 | PASS | Maple과 같은 projection shape, 합성 가중치 |
| QKV edge 4 variants | PASS | M33/17/15, N5, K512, code3, zero scale, byte offset2/6/10, row tails |
| Source collector | PASS | mock source 포함, build directory 거부 |
| Unexecuted fixture checker | PASS | CPU 정답만 있고 실행 결과가 없으면 실패 |
| 원본 SPIR-V + Vulkan + A750 | 미검증 | 도구/장치 없음 |
| 실제 Maple GGUF/logits/PPL | 미검증 | 전체 모델/기존 backend 미연결 |
| 기존 llama.cpp보다 빠른가 | 미측정 | 여기서는 속도 향상을 주장하지 않음 |

모든 합성 layer fixture에서 router expert IDs는 CPU reference와 동일했다. 최종 MoE output의 일부 결과:

| fixture | max absolute error | NMSE |
|---|---:|---:|
| F32 N13 | 9.12696e-8 | 2.41778e-13 |
| Q8 N13 | 6.70552e-8 | 3.34350e-14 |
| Q8 N184 | 2.15471e-5 | 3.50528e-10 |
| Q8 N2048 | 3.37921e-5 | 1.11951e-10 |
| Q8 Maple-shape N1 | 4.17233e-7 | 4.26433e-14 |

기본 checker tolerance는 `rtol=3e-4, atol=3e-4`다. Q8 N2048의 per-expert down output max error는 `2.77460e-4`다. CPU와 소프트웨어 셰이더의 activation 함수/rounding의 작은 차이가 후속 Q8 반올림 경계에서 커질 수 있으므로, 이를 bit-exact라고 표시하지 않았다. 이 해석은 가능한 수치 원인이며 모든 차이의 내부 원인을 추적해 확정한 것은 아니다.

Q8 비교 기준은 동일 Q8g32 규약의 CPU oracle이다. F32 모델 대비 양자화 품질 손실을 측정한 결과가 아니다. Q/K/V가 일부 fixture에서 bit-exact여도 전체 모델/다른 backend의 bit-exact를 뜻하지 않는다.

## 보조 GLSL 변환의 범위

`set=0` 제거, push constant→동일 scalar offset의 UBO, include 확장, Vulkan integer-dot intrinsic→네 signed byte 정수 곱/합으로 치환했다. Shared memory, workgroup barriers, atomic counter/bucketing, projection, routing은 실제 셰이더 코드로 컴파일/실행했다.

이 검증은 index/barrier-uniformity/수치 알고리즘의 많은 오류를 잡을 수 있지만 Vulkan pipeline layout·SPIR-V capability·Vulkan 메모리 모델·A750 compiler/occupancy를 증명하지 않는다. Mesa 실행도 data race 부재를 형식 증명하는 것은 아니다.

## 재현과 증거

원시 출력은 `evidence/tests/`에 있다. `evidence/validation-cases.json`에는 사용한 fixture 옵션과 execution provenance를 저장했다. 대형 raw fixture는 배포 ZIP에 포함하지 않았다. `tools/make_fixture.py`의 seed와 옵션으로 재생성한다.

`execution.json`이 없거나 control dump가 fixture manifest보다 오래되면 checker가 실패한다. 제공 실행기들은 재실행 시작 시 해당 plan이 선언한 이전 output을 지워 실패 후 남은 파일을 성공으로 오인하지 않게 한다. 이는 암호학적 tamper-proof 검증이 아니라 개발 중 stale-output 방지 장치다.
