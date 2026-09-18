# Maple TQ2 Vulkan Lab v0.1

**상태: 1–6번 최적화의 독립 실험 커널 + 실행/검증 도구. llama.cpp 통합 패치나 교체 DLL이 아니다.**

사용자가 보낸 `build-sycl-vulkan(1).zipx`에는 원본 소스 대신 빌드 산출물이 있었다. 확인한 build-info는 `10993 / da0b59a62`, GGML은 `da0b59a62-dirty`이며 원본 경로는 `C:/AI/llama-src`다. 이 dirty 소스의 graph/backend 인터페이스를 보지 못한 상태에서 upstream으로 덮어쓰지 않았다. 제공 코드는 기존 파일을 변경하지 않는 독립 디렉터리에서 실험한다.

**현재 확인한 것:** C++/NumPy 기준 테스트, Mesa 소프트웨어 OpenGL로 변환한 셰이더의 컴파일·실행·수치 비교.

**확인하지 못한 것:** 원본 GLSL→SPIR-V 컴파일, `vk_replay.cpp` 컴파일, Vulkan validation, A750 실기, native integer-dot instruction, 실제 Maple GGUF의 logits/PPL, llama-server 연결 및 속도 향상. 이 환경에는 Vulkan SDK와 Vulkan 장치가 없었다. 아래 Windows 명령은 실기 검증용 절차이며, 여기서 실행 성공한 명령이 아니다.

## 1. 구현 범위

| 번호 | 내용 | 구현 | 중요한 경계 |
|---|---|---|---|
| 1 | expert bucketing + persistent grouped matmul | count → prefix → scatter → tile job 생성 → atomic task cursor | 작은 decode에는 direct 경로. 기존 llama.cpp보다 빠르다는 결과는 없음 |
| 2 | up + gate fusion | 두 TQ2 행렬을 읽어 같은 activation tile로 계산하고 clamped SwiGLU까지 출력 | up/gate 가중치를 서로 같은 값으로 공유하는 것이 아님 |
| 3 | TQ2 × Q8 INT8 dot | VRAM TQ2 유지, packed INT8 unpack, `dotPacked4x8EXT` | Q8g32는 추가 양자화이며 기본 OFF. native instruction 실기 미검증 |
| 4 | unpack-once/reuse | `32 output rows × 4 assignments × K256` 타일의 unpack을 SLM에서 공유 | 데이터 재사용 증가와 SLM/barrier 비용의 교환. A/B 제공 |
| 5 | QKV fusion | 3개의 기존 weight buffer를 virtual row-concat으로 한 dispatch에 계산 | GGUF 변경/영구 복제 없음. 독립 테스트의 split copy는 통합 시 strided view로 대체해야 함 |
| 6 | router fusion | FP32 parallel router GEMV + softmax/top-k/renormalization 단일 커널 | projection까지 한 WG로 합치는 `--full-router`도 별도 실험 옵션. 기본 아님 |

MLX Maple의 expert sort/unsort, up/gate·QKV 결합, FP32 routing, 정확한 SwiGLU clamp를 참고했다. **MLX 전체 구현의 bit-exact 포팅은 아니다.** Persistent queue와 Q8g32/INT8 경로는 이 패키지의 별도 설계다. MLX affine 2-bit와 GGML TQ2는 다른 포맷이므로 packed bytes를 서로 혼용하지 않는다.

## 2. 수치·레이아웃 계약

TQ2 한 블록은 `256 values = 64 packed bytes + FP16 scale 2 bytes`다. 요소 i의 byte는 `32*(i/128)+i%32`, shift는 `2*((i%128)/32)`이다. code는 `0,1,2,3 → -1,0,+1,+2`로 해석한다. native ternary 변환기에서 code3이 안 나온다는 사실에 커널을 의존시키지 않는다.

매 256-element 블록의 scale을 따로 읽는다. row 전체 scale이 같다고 가정하지 않는다. 블록 stride66 때문에 32-bit 경계에서 어긋나는 접근과 4-byte padding을 처리했다. weight descriptor 기준 byte offset은 짝수여야 한다.

기본 F32 경로는 activation을 추가 양자화하지 않는다. 다만 기존 ggml/MLX의 reduction 순서·F16/BF16 중간 rounding까지 같다는 뜻은 아니다.

Q8g32 경로는 32개 activation당 FP32 absmax/127 scale 하나와 signed INT8 값을 사용한다. 이것은 **ggml Q8_K/Q8_1의 버퍼 레이아웃이 아니다.** INT8 누산 뒤 weight/activation scale을 FP32로 곱한다. 이 경로의 테스트는 같은 Q8 수치 규약에 대한 정확성이지 F32 모델과 품질 동등성을 뜻하지 않는다.

Maple MoE activation은 다음과 같다.

```text
silu(min(gate, 7)) * clamp(up, -7, 7)
```

Gate에는 아래쪽 -7 clamp가 없다. Down projection 입력은 token이 아니라 **선택 expert별 assignment activation**이다. Expert 출력은 원래 top-k slot 순서로 복원한 뒤 순서가 고정된 FP32 reduce를 한다.

Router는 전체 expert에 FP32 softmax를 적용한 뒤 그 확률에서 top-k를 고르고 합+1e-20으로 재정규화한다. 기본 tie policy는 낮은 ID 우선, 내림차순이다. `--tie-high --ascending`도 있다. 실제 backend와 비교할 때 tie/order를 먼저 맞춰야 하며, near-tie의 reduction 차이를 무조건 버그라고 판정하지 않는다.

## 3. CPU 테스트

Python 3.10 이상, NumPy, C++17 컴파일러와 CMake가 필요하다.

```powershell
python -m pip install -r requirements.txt
cmake -S . -B build-cpu
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
python -m unittest discover -s tests -p test_reference.py -v
```

상세 결과와 한계는 `docs/VALIDATION.md`, 원시 로그는 `evidence/tests/`에 있다. 합성 데이터는 패키지에 대량 포함하지 않고 seed로 재생성한다.

## 4. Windows/A750의 독립 Vulkan 실험

VS 개발자 터미널 또는 C++ 컴파일러가 잡힌 셸에서 실행한다. 최신 Vulkan SDK의 headers/library, `glslc`, 선택적으로 `spirv-val`이 필요하다. Ninja가 없으면 `-G Ninja`를 생략하고 생성된 executable의 `Release` 하위 경로를 사용한다.

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMAPLE_BUILD_VULKAN=ON
cmake --build build
.\build\maple-tq2-vk.exe --list
```

아래 `$Device`는 `--list`에서 확인한 A750 번호로 설정한다. **기존 llama-server DLL을 바꾸는 명령은 없다.**

```powershell
$Device = 0
python tools/make_fixture.py --out fixtures/f32 --tokens 13
.\build\maple-tq2-vk.exe fixtures/f32/run.plan build/shaders --device $Device --validation
python tools/check_fixture.py fixtures/f32
```

프로그램 실행 종료와 correctness PASS는 별개다. 마지막 checker까지 통과해야 한다. Validation layer가 설치되어 있지 않으면 `--validation`은 명시적으로 실패한다. 진단을 위해 layer 설치를 권장한다.

추가 activation quantization 경로는 별도 fixture로 실행한다.

```powershell
python tools/make_fixture.py --out fixtures/q8 --tokens 13 --q8
.\build\maple-tq2-vk.exe fixtures/q8/run.plan build/shaders --device $Device --validation
python tools/check_fixture.py fixtures/q8
```

Q8 실행은 `shaderIntegerDotProduct` 기능이 없으면 중단한다. 기능 지원과 실제 하드웨어 가속 여부는 별개이므로 출력되는 `Packed signed INT8 dot accelerated`도 확인한다. 이 경로는 packed INT8 dot이며 **XMX cooperative-matrix 경로를 구현했다는 뜻은 아니다.**

Maple과 같은 projection 크기의 단일 layer 합성 실험:

```powershell
python tools/make_fixture.py --out fixtures/maple-shape --tokens 1 --hidden 2048 --ffn 512 --q8
.\build\maple-tq2-vk.exe fixtures/maple-shape/run.plan build/shaders --device $Device --validation
python tools/check_fixture.py fixtures/maple-shape
```

합성 fixture의 device buffer payload는 약 201.67MiB다. 업로드/다운로드 staging, allocator alignment, pipeline 메모리는 별도다. 전체 Maple 모델의 메모리 사용량이 아니다.

## 5. A/B 옵션

옵션들은 `make_fixture.py`에 준다. 같은 seed/shape/precision에서 한 변수만 바꾸고, 각각 checker 통과 후 `timing.csv`를 비교한다.

| 항목 | 옵션 |
|---|---|
| expert 경로 | `--route direct`, `--route bucket`, `--route auto` |
| persistent | `--persistent`, `--no-persistent` |
| resident WG 수 | `--workgroups 32`, `64`, `128`, `256` |
| up/gate fusion | `--fuse-upgate`, `--no-fuse-upgate` |
| unpack 재사용 | `--reuse`, `--no-reuse` |
| QKV fusion | `--fuse-qkv`, `--no-fuse-qkv` |
| 추가 INT8 activation | `--q8` 또는 생략(F32) |
| router 전체 fusion | `--full-router` 또는 생략(분리 projection+fused selection) |
| tie/order | `--tie-high`, `--ascending` |

`auto`의 `assignments >= 64`와 기본 persistent WG128은 **실험 시작값**이지 A750 튜닝 결과가 아니다. Direct/bucket, persistent/static, SLM 재사용의 최선은 batch와 expert histogram에 따라 달라질 수 있다. 모든 fusion을 켜는 것이 반드시 가장 빠르지 않다.

WG를 2D로 펼쳐 큰 Q8 quantization dispatch의 x축 65,535 한계를 처리했다. 하나의 buffer 또는 상대 addressing이 4GiB를 넘거나 device storage range를 넘으면 분할이 필요하다. Router는 최대256 experts/top-k8이다. 이 prototype은 generated plan을 신뢰하는 개발 도구이며, 임의의 외부 plan을 안전하게 실행하는 보안 sandbox가 아니다.

측정값은 독립 synthetic pipeline의 stage 시간이다. 기존 llama.cpp와의 end-to-end 비교도, PPL/logit 평가도 아니다. 초기 version은 정확성 추적을 위해 dispatch 사이에 넓은 barrier를 사용한다. 실기 최적화 단계에서 정확한 resource dependency로 좁힐 수 있다.

## 6. 기존 dirty llama.cpp에 연결하기 위한 소스 수집

첨부 build의 `compile_commands.json`에 나타난 실제 원본 경로를 사용한다.

```powershell
python tools/collect_llama_source.py --root "C:\AI\llama-src" --out "llama-actual-source.zip"
```

빌드 폴더를 다시 선택하면 필요한 원본 파일을 확인한 뒤 오류로 중단한다. 현재 working tree의 수정·미추적 소스도 포함하고 `.git`, build 폴더, 모델, DLL/OBJ 등은 제외한다. 소스의 설정/비밀정보는 공유 전에 검토해야 한다. 기존 출력 ZIP은 덮어쓰지 않는다.

실제 ggml graph fusion, allocator/view/stride, request별 scratch lifetime, SYCL/oneDNN과의 routing, CPU fallback 연결 지점은 `docs/INTEGRATION.md`에 정리했다. **이 연결은 이번 패키지에서 수행되지 않았다.**

## 7. Linux 보조 검증

Mesa EGL/OpenGL 4.5 compute가 설치된 환경 전용이다. Vulkan binding/push-constant 선언을 OpenGL로 변환하고 native packed dot을 네 signed integer 곱으로 치환한다.

```bash
python tools/make_fixture.py --out fixtures/software --tokens 13 --q8
python tools/glsl_software_oracle.py fixtures/software
python tools/check_fixture.py fixtures/software
python tests/test_software_edges.py
```

`execution.json`은 이를 `Mesa EGL/OpenGL software oracle`, `vulkan_tested=false`, `native_integer_dot_tested=false`로 기록한다. 보조 테스트를 GPU/Vulkan PASS로 혼동하지 않는다.
