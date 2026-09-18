# Maple W2A16 + native W2A8 XMX 통합 실험 v0.2

## 결과물의 범위

기존 v0.1 W2A16 소스에 **native signed INT2 × signed INT8 DPAS** 초안을 추가했다.
두 경로를 한 바이너리에서 같은 device/context/queue, 같은 weight/activation/expert ID로 비교한다.
**완전한 SYCL Maple backend나 llama-server 패치가 아니다.** Vulkan/oneDNN interop과 서버 설정은 변경하지 않는다.

**이 환경에서 검증:** 기존 A16 CPU reference 검사, 신규 A8 CPU reference 검사,
GGUF 추출기·통합 실행기 Python tests, ASan/UBSan. 정확한 결과는 `validation/`에 있다.

**미검증:** Intel SYCL 컴파일, A750에서 실행한 수치 결과·device ISA·속도, Windows PowerShell 실행,
Vulkan↔SYCL 연결 비용과 end-to-end Maple 품질/TG. 이 컨테이너에는 icx/icpx와 A750이 없다.
GPU 컴파일 또는 probe 실패 시 중단하고 로그를 보존한다. CPU PASS를 GPU PASS로 해석하지 않는다.

## 가장 빠른 실행

oneAPI + MSVC 환경이 초기화된 PowerShell에서 압축 해제한 디렉터리로 이동한다.
설치나 영구 환경변수 변경은 하지 않는다. 측정 중 다른 GPU 작업은 중지한다.

```powershell
.\scripts\build_and_test.ps1 -Device A750 -Suite quick `
  -Model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf"
```

첫 실행은 GPU 소스 A16/A8 각각을 한 번 컴파일하고 두 테스트 바이너리에 재사용한다.
통합 실행기는 한 case 안에서 모든 variant를 같은 프로세스·queue에서 비교한다.
**서로 다른 shape case는 별도 프로세스**이므로 First/JIT는 각 case마다 따로 기록한다.

`-Model`을 생략해도 synthetic smoke와 Maple 형상 비교가 가능하다.
실제 가중치를 쓰는 경우 layer 0의 gate/up/down tensor만 별도 `.mw2`로 추출한다.
원본 GGUF에는 쓰지 않는다. 다른 layer는 `-Layer 7`처럼 지정한다.

반복할 때 재빌드할 필요는 없다.

```powershell
.\scripts\build_and_test.ps1 -NoBuild -Device A750 -Suite full `
  -Model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf" -Repeats 28
```

`quick`은 smoke 4 case + 실제 Maple gate/up와 down의 Q1 2 case, 총 6 case다.
`full`은 Q4/Q8, split-K, outlier, local-size 대조군을 포함한 총 17 case다.
CPU/ISA probe/수치 검사 실패는 성공으로 숨기지 않고 suite를 실패 처리한다.

출력은 `build/results/<timestamp>-<suite>/`에 저장된다. 이미 결과가 있는 디렉터리는 덮어쓰지 않는다.

```text
all_summary.csv                 전체 case·variant 집계
suite_status.json               완료 여부와 실패 원인
native-int2-probe.log            signed INT2/INT8 probe
<case>/manifest.txt              형상·옵션·입력 hash·device·driver·메모리 정보
<case>/samples.csv               round별 First/Warmup/Repeat 시간
<case>/summary.csv               variant별 median/p95/정확성
<case>/numerical.csv             두 routing snapshot의 projection별 오차
```

## 세 개의 주 비교 경로

| 경로 | Weight | Activation | 계산 |
|---|---|---|---|
| W2A16 | TQ2 또는 signed2 재배열 | F16 또는 F32→F16 | FP16×FP16 DPAS, FP32 누산 |
| W2A8 staged | 동일 | GPU에서 매 호출 그룹별 A8 생성 | INT2×INT8 DPAS, INT32 부분합, FP32 재결합 |
| W2A8 fused | 동일 | dot 커널 내부에서 A8 생성 | 동일 native INT2 DPAS |

**native W2A8은 DP4A 대체 표기가 아니다.** 소스는 다음 명령 API를 명시적으로 요청한다.

```cpp
ia = xmx::dpas<8, 1,
    int32_t, int32_t, uint32_t, int8_t,
    xmx::dpas_argument_type::s2,
    xmx::dpas_argument_type::s8>(ia, packed_b, a8);
```

DG2 기준 B는 `simd<uint32_t,16>`(K32×N8의 2bit = 64 bytes),
A는 `simd<int8_t,32>`, 누산기는 `simd<int32_t,8>`이다.
최종 생성 ISA는 로컬 도구로 확인해야 한다. `requested_*` 로그는 관측한 ISA가 아니다.
Probe도 수학적 결과를 확인하는 것이며 disassembly를 대신하지 않는다.

## Weight layout을 통일했다

`native`는 원래 표준 GGML TQ2_0의 66-byte/256-weight block을 직접 읽는다.
`W2A8 native`는 register에서 signed2 VNNI로 배치한다.

`s2tile8`은 8행×256개의 weight를 **528 bytes 그대로** 저장하는 lossless 재배열이다.
A16과 A8이 **같은 s2tile8 GPU 버퍼를 공유**한다. INT8/FP16 weight 전체 사본은 만들지 않는다.

```text
TQ2 코드 0,1,2       → 값 -1,0,+1
signed INT2 비트 3,0,1 → 값 -1,0,+1

INT2 B matrix K32×N8:
  DWORD index = (k / 16) * 8 + output_row
  bit offset  = 2 * (k % 16)
```

원래 weight scale은 여전히 256개당 F16 하나다. block마다 다른 scale과 음수 scale도 보존한다.
TQ2 reserved code 3 및 nonfinite scale은 추출/load 검사에서 거절한다.
INT2 probe는 하드웨어 부호 확장을 검사하려고 -2까지 쓰지만, 모델 weight에는 -2를 허용하지 않는다.

기존 v0.1의 `tile8`은 **다른 패킹**이다. 두 이름을 혼용하지 않는다.
legacy `maple-w2-bench`는 기존 `native/tile8`을 계속 지원하고, 통합 비교기는 `native/s2tile8`을 사용한다.

기본 comparator는 두 layout을 함께 측정하므로 해당 **추출 tensor의 두 GPU 사본**을 잠시 보유한다.
`manifest.txt`의 `weight_bytes_resident_in_comparator`에 표시한다.
이는 production에서 전체 모델을 두 벌 상주시킬 것을 권하는 설계가 아니다.
한 layout만 측정하려면 `--layouts s2tile8` 또는 `--layouts native`를 지정한다.

## Activation group은 weight group과 독립적이다

A8 그룹 G는 **32 / 128 / 256**이다. TQ2 weight group을 바꾸지 않는다.
TQ2-256도 A8-G32를 사용할 수 있으며, Q2 weight 포맷으로 변환할 필요가 없다.

```text
d_x = max(abs(x)) / 127
q_x = round_to_nearest_even(clamp(x / d_x, -127, +127))
```

zero group은 `d_x=1, q_x=0`으로 표현한다. 극히 작은 scale은 float 최소 normal 값으로 하한 처리한다.
입력이 NaN/Inf이면 GPU invalid flag를 남기며 결과를 채택해서는 안 된다.
통합 비교는 A16과 조건을 맞추기 위해 finite·FP16 유한 범위 이내 입력만 받는다.

이 A8은 **별도 실험용 symmetric 그룹 포맷**이다. `GGML Q8_1`의 binary layout이나 scale/sum 규칙과 동일하지 않다.
FP32 activation scale을 사용하며, FP16-scale Q8_1과 수치적으로 같다고 주장하지 않는다.

```text
y = Σ_group [ weight_scale(block256) × activation_scale(group G)
              × INT32_dot(ternary_codes, quantized_activation) ]
```

G32에서도 서로 다른 scale을 하나로 합치지 않는다.
최대 INT32 부분합은 256×127=32,512로 이번 strict-ternary 그룹 범위 안에서는 overflow하지 않는다.

**staged:** input row마다 한 번 quantize한다. gate/up 및 선택 expert들은 같은 q/scales를 읽는다.
**fused:** output tile마다 quantize를 반복하는 대신 별도 quantize launch와 q scratch 왕복을 피한다.
paired gate/up에서는 quantized activation을 두 projection이 공유한다.
어느 쪽이 빠른지는 `submit_wait_median_us`로 판단한다.

## 수치 검사를 세 부분으로 분리한다

1. **GPU vs GPU가 실제 사용한 A8 code/scale의 CPU gold:** INT2 패킹·expert 주소·scale·누산 검사.
2. **GPU quantizer vs CPU quantizer:** scale과 code를 비교한다. division 오차로 half-integer 근처에서
   발생한 ±1 code 차이는 `q_near_tie`로 별도 보고하며, 그 밖의 차이는 실패다.
3. **A8 gold vs 원래 F32 activation gold:** activation 양자화만의 오차. A16도 F16 반올림 오차를 따로 보고한다.

Fused 경로의 q/scales 캡처는 **시간 측정 후 numerical pass에만** 켠다.
시간 측정에서는 fused의 q/scales를 메모리에 쓰지 않는다. Invalid flag 기록은 유지한다.

Kernel 통과 조건은 finite, aggregate NMSE≤1e-6, max_abs/reference_RMS≤0.005다.
이 조건은 기존 v0.1의 arithmetic 검사와 같다. **모델 품질 기준은 아니다.**
`precision_only_nmse`는 정보로 보고하며, 낮은 MSE만으로 agent 정확성이나 수락률이 보장된다고 하지 않는다.

검사는 첫 routing snapshot과 마지막 snapshot을 사용한다. 생성된 첫 snapshot에는
최종 expert ID, 0, 1...을 강제로 포함해 expert offset 경계를 검사한다.
매 timing iteration의 출력을 읽어서 검사하는 방식은 아니다.

## Wall time 측정 범위

```text
quant_median_us        staged activation quantize; fused에서는 0(비용은 Main에 포함)
main_median_us         dot + group rescale 등 Main 커널
reduce_median_us       선택적 split-K 결과 합산
span_median_us         첫 GPU event 시작 → 마지막 event 끝 (CSV: event_span_median_us)
submit_wait_median_us  host enqueue 직전 → completion wait 직후
submit_wait_p95_us     같은 범위의 p95
first_submit_wait_us   First/JIT 별도
```

staged에는 **매 호출마다 quantize 비용이 포함**된다. 미리 quantize한 뒤 dot 시간만 비교하지 않는다.
A16/A8은 같은 input 및 expert ID를 매 round 사용하고, variant 순서를 회전·반전한다.

최초 weight upload/repack, host readback, CPU gold 계산, Vulkan↔SYCL handoff,
attention, router, SwiGLU, sampling은 포함하지 않는다. **server TG가 아니다.**

`--include-fma 1`의 FMA는 기존 Vulkan shader가 아니라 v0.1의 단순 SYCL 대조 커널이다.
이를 이겼다고 현재 Vulkan보다 빨라졌다고 주장하면 안 된다.

`--scrub-mib 64`는 측정 밖에서 별도 메모리 영역을 쓰는 cache-control 대조군이다.
cache cold 보장은 없으며 실제 24층 모델 trace를 대신하지 않는다.
`--id-mode rotate` 역시 라우팅을 바꿀 뿐 전체-model weight streaming을 재현하지는 않는다.

## 실제 activation 캡처로 비교

통합 실행기의 `-Model`은 **weight만 실제 값**으로 바꾼다. 기본 activation과 routing은 여전히 합성이다.
캡처 데이터를 이용한 직접 실행:

```powershell
.\build\maple-w2-compare.exe `
  --weights .\gate.mw2 --weights2 .\up.mw2 `
  --x-file .\gate-x.f32 --ids-file .\gate-ids.i32 `
  --tokens 1 --topk 8 --per-selection 0 --input f32 `
  --layouts native,s2tile8 --groups 32,128,256 --a8-modes staged,fused `
  --split 1 --repeats 28 --out .\build\results\captured-gate-up
```

입력 파일은 little-endian F32, ID는 little-endian I32.
Gate/up X `[Q,K]`, down X `[Q,topk,K]`, ID `[Q,topk]`다.
F16 원본은 동일 값의 F32로 캡처하고 `--input f16`으로 다시 올린다.
이번 API는 Q1..8의 독립 matvec들이다. **prefill용 expert GEMM을 구현한 것이 아니다.**

## 빌드만 / CPU 검사만

```powershell
.\scripts\build_windows.ps1
.\build\maple-w2-compare.exe --device A750 --probe-only 1
```

Linux oneAPI:

```bash
cmake -S . -B build-sycl -G Ninja -DMAPLE_BUILD_SYCL=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl
python3 tools/run_suite.py --exe build-sycl/maple-w2-compare --suite quick --out build/results/linux
```

CPU-only (GPU 컴파일을 의미하지 않음):

```bash
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

## 서버 통합 전 계약

`enqueue_a8()`와 기존 `enqueue()`는 외부 queue, dependency event, persistent scratch를 받는다.
호출 내부에서 host wait나 allocation을 하지 않는다. 반환 completion 이전에 scratch를 재사용·해제하지 않는다.
staged quant event → dot → optional reduce는 dependency로 연결된다.
두 경로가 사용하는 모든 pointer는 같은 context의 USM 또는 검증된 import pointer여야 한다.

기존 `integration/ggml_decode_adapter.hpp`는 A16의 제한적 Q1 view adapter로 보존했다.
**A8 정밀도 허용 여부·quant workspace·event 수명 관리를 연결하는 서버 adapter는 별도 작업이다.**
`.wait()`를 삭제하거나 raw Vulkan 주소를 USM으로 cast하는 패치를 넣지 않았다.

## 참고한 원시 명세

- Intel DPAS ISA: https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
- Intel ESIMD DPAS API: https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/xmx/dpas.hpp
- 비교 출발점: 사용자 대화의 `maple-w2a16-xmx-v0.1-source.zip`.

`SOURCE_PROVENANCE.json`에 입력 zip SHA256과 변경 범위를 기록했다.
