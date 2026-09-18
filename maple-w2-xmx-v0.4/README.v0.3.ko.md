# Maple W2 XMX v0.3 — 연결된 MoE 블록의 A16/A8 비교

## 검증 상태

**기존 v0.2는 사용자가 제공한 A750 `run.log`에서 빌드·GPU 수치 검사에 성공했다.**
6개 case / 85개 variant summary가 PASS이며, signed INT2/INT8 probe 36 cases / 288 outputs도 PASS다.
컴파일러는 oneAPI 2026.1.1, SYCL driver 문자열은 `1.15.39183+4`다.
원문과 집계는 `evidence/`에 보존했다.

**새 v0.3은 여기서 CPU reference·Python orchestration·ASan/UBSan 검사를 통과했다.**
그러나 작성 환경에는 oneAPI 컴파일러와 A750, Windows PowerShell이 없어
**새 SYCL 소스 컴파일·GPU 실행·PowerShell 실행은 아직 검증하지 못했다.**
v0.2의 GPU 성공을 v0.3의 성공으로 간주하지 않는다.

이 패키지는 standalone 실험 코드다. 기존 `ggml-vulkan.dll`, 서버, 모델 파일, 실행 옵션을
수정하지 않는다. Vulkan-native 구현이나 Maple 전체 SYCL backend 포트가 아니다.

## 왜 이 버전을 추가했나

실제 layer-0 weight + 합성 activation/라우팅의 v0.2 결과:

| 구간 / s2tile8 공통 layout | A16 wall 중앙값 | A8-G32 staged | 시간 감소 |
|---|---:|---:|---:|
| Gate/up pair, Q1 | 328.4us | 228.6us | 30.39% |
| Down, Q1 | 225.2us | 200.2us | 11.10% |

G128 staged는 각각 228.8us / 197.7us로, G32보다 확실히 우월하다고 고정하기에는 차이가 작고
해당 입력의 양자화 오차도 더 컸다. G32를 기본 대조군으로 유지하고 G128/G256도 함께 비교한다.

v0.2의 fused matvec은 출력 타일마다 activation 양자화를 반복하는 방식이다.
이번 v0.3의 `gluquant`는 **다른 fusion**이다: SwiGLU에서 생성하는 hidden activation을
한 번 양자화하고, down 커널이 그 q/scale을 그대로 소비한다.

## 새 실행 경로

한 MoE 층에 대해 동일 queue와 persistent workspace에서 세 경로를 비교한다.

```text
A16
    gate/up DPAS → clipped SwiGLU → down DPAS → routing-weight 합산

A8-separate
    input quant → gate/up INT2×INT8 DPAS → clipped SwiGLU
                → hidden quant → down INT2×INT8 DPAS → routing-weight 합산

A8-gluquant
    input quant → gate/up INT2×INT8 DPAS → clipped SwiGLU + hidden quant
                → prequantized down INT2×INT8 DPAS → routing-weight 합산
```

모든 A8 경로의 gate/up은 staged input quant를 사용한다. 이미 양자화된 입력을 선택 expert와
두 projection이 공유한다. `prequantized`는 **이번 호출의 producer 결과를 재사용**하는 것이며,
이전 토큰의 activation을 재사용하는 방식이 아니다.

`clipped SwiGLU`는 다음 식이다. GPT-OSS의 alpha=1.702 또는 up+1 변형이 아니다.

```text
SiLU(min(gate, clamp)) * clamp(up, -clamp, +clamp)
clamp 기본값 = 7
```

원래 TQ2 weight scale/256은 모두 보존한다. Input A8 group과 hidden A8 group은 독립적이다.
G32/G128/G256을 지원하며, 예를 들어 gate input G32 + down input G128도 시험한다.
A8 scale은 FP32인 실험 포맷으로, GGML Q8_1과 binary-compatible하지 않다.

### 비동기 실행과 대조군

기본 `enqueue_moe()`는 explicit dependency event를 연결하며 **중간 host wait를 하지 않는다.**
전체 체인의 completion event를 반환한다. Input upload·workspace 할당·CPU reference 계산은
호출 외부에 둔다. In-order와 out-of-order queue를 모두 시험한다.

같은 체인에 gate/up 뒤, SwiGLU 뒤, down 뒤 **3회 host wait를 고의로 삽입한 대조군**도 실행한다.
이것은 SYCL 내부의 동기식 제출 영향 측정이지, Vulkan↔Level Zero 외부 semaphore/import 비용을
재현한 테스트가 아니다. 실제 runtime이 암묵적으로 block하지 않는다는 보증도 아니므로
`enqueue_return_us`와 GPU event gap을 기록한다.

### Device query hot path

A750/A770 검증 결과를 device identity별로 캐시한다. v0.2처럼 모든 enqueue에서 device name,
vendor, FP16 support를 다시 조회하지 않게 했다. 지원 장치 검사를 생략하는 것은 아니다.
효과는 아직 측정하지 않았으며 다음 진단 switch로 기존 반복 조회를 재현할 수 있다.

```powershell
$env:MAPLE_W2_DEVICE_CHECK_EVERY_CALL = '1'
# 새 프로세스로 비교한 뒤, 기본 정책 복원:
Remove-Item Env:MAPLE_W2_DEVICE_CHECK_EVERY_CALL
```

## 가장 빠른 새 테스트

v0.2 디렉터리를 덮어쓰지 말고 별도 `maple-w2-xmx-v0.3` 폴더로 압축을 푼다.
그 폴더에서 다음 한 줄을 실행한다.

```powershell
.\build_and_test.cmd -MoeOnly -Suite quick -Device A750 -Model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf"
```

`.cmd`는 이미 초기화된 compiler를 우선 사용하며, 없으면 설치된 oneAPI의 `setvars.bat`을
자식 프로세스 안에서 호출한다. 시스템 설치나 `setx`, 영구 ExecutionPolicy 변경을 하지 않는다.
`python` 대신 venv를 지정하려면 `-Python "C:\openvino-qwen\Scripts\python.exe"`를 추가한다.

`-MoeOnly`는 v0.2의 개별 matvec GPU 스윕을 건너뛰는 옵션이다. CPU reference 검사와 새 MoE
smoke/실제 weight 테스트는 수행한다. 이미 검증한 `gate.mw2 / up.mw2 / down.mw2` 폴더를
재사용하려면 `-Model` 대신 `-CapsuleDir "기존 결과의 weights 폴더"`를 지정한다.

기존 개별 matvec과 새 체인을 모두 한 번에 회귀 시험하려면 `-MoeOnly`를 빼면 된다.

```powershell
.\build_and_test.cmd -Suite quick -Device A750 -Model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf"
```

### 빌드 경로 보완

성공한 사용자 로그에 맞춰 `icpx.exe`를 우선 선택하고 `-fexceptions -fcxx-exceptions`를 명시했다.
`clang_rt.builtins-x86_64.lib`는 선택 compiler의 `-print-resource-dir`에서 검색해 연결한다.
찾지 못했을 때 실제 설치 경로를 `-Builtins`로 지정할 수 있다. oneAPI 버전 경로를 고정하지 않는다.

GPU 소스 A16/A8/MoE를 각각 한 번 컴파일한 object로 세 비교기를 링크한다.
기존 matvec용 `maple-w2-compare.exe`, 호환용 `maple-w2-bench.exe`,
새 `maple-moe-compare.exe`가 생성된다.

### 재빌드 없이 확장

```powershell
.\build_and_test.cmd -NoBuild -MoeOnly -Suite full -Device A750 -Model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf" -Repeats 28
```

| MoE suite | Case 수 | 내용 |
|---|---:|---|
| smoke | 2 | 작은 Q1 + Q4 out-of-order, 합성 weight |
| quick | 3 | smoke 2개 + Maple 실제 shape Q1 |
| full | 7 | quick + Maple Q4, split-K, G32/H128, outlier Q8 |

보통 case당 A16 + A8-separate 3groups + A8-gluquant 3groups = 7variants를
async/waited 각각 실행해 14variants다. G32/H128 전용 case는 6variants다.
Full MoE suite의 합계는 90variants다. 기존 tensor suite는 별도다.

Maple test도 `-Model`/`-CapsuleDir`을 주지 않으면 합성 weight다.
모델을 지정해도 **기본 activation·expert IDs·routing weights는 합성**이며 실 decoder trace가 아니다.

## 실제 activation·라우팅 재생

`maple-moe-compare.exe`를 직접 호출할 때:

```powershell
.\build\maple-moe-compare.exe --device A750 --gate .\gate.mw2 --up .\up.mw2 --down .\down.mw2 --tokens 1 --topk 8 --x-file .\x.f32 --ids-file .\ids.i32 --routes-file .\routes.f32 --layout s2tile8 --groups 32,128,256 --host-waits both --out .\build\results\captured-moe-001
```

파일은 little-endian raw 형식이다. `x.f32`는 norm 이후 gate/up 공통 입력 `[Q,K]`,
`ids.i32`는 `[Q,topk]`, `routes.f32`는 이미 정규화된 선택 expert별 가중치 `[Q,topk]`다.
K/H/expert 수는 capsule에서 읽는다. 각각의 GGUF layer가 서로 일치해야 한다.
Route는 finite/nonnegative이며 token별 합이 1(허용차1e-4)인지 확인하고, 임의로 재정규화하지 않는다.
원래 routing policy가 다르면 이 API 계약을 먼저 맞춰야 한다.

원래 layout도 같은 test에서 확인하려면 `--layout native`를 사용한다. 기본 chain benchmark는
선택 layout 하나만 GPU에 올려 **native+s2tile8 weight 이중 상주를 피한다.**
Repack/upload 시간은 steady-state timing 밖에 둔다.

## 결과 읽는 순서

상위 결과 폴더는 `build/results/<timestamp>-<suite>/`다.

```text
run.log                           전체 실행 transcript
moe/all_moe_summary.csv            각 MoE case/variant 요약
moe/suite_status.json              완료·실패·capsule hash
moe/<case>/numerical.csv           단계별 구현 오차 + 전체 체인 오차
moe/<case>/summary.csv             median/p95 및 A16 대비 배율
moe/<case>/samples.csv             반복별 wall/GPU time
moe/<case>/stages.csv              quant/gate-up/GLU/down/sum event
moe/<case>/manifest.txt            device·입력 출처·명시적 버퍼 크기
```

### 시간 정의

- `wall_median_us`: CPU enqueue 시작 → 전체 chain completion 대기 종료.
- `enqueue_return_median_us`: enqueue 시작 → enqueue 함수 반환. waited 모드는 고의 host wait도 포함.
- `tail_wait_median_us`: 반환 뒤 completion을 기다린 시간. GPU 계산 시간과 중복되므로 별도 overhead로 더하지 않는다.
- `kernel_sum_median_us`: 개별 GPU kernel event duration의 합.
- `gpu_span_median_us`: 첫 GPU kernel 시작 → 마지막 kernel 종료.
- `gpu_gap_median_us`: GPU span - kernel sum. 직렬 의존 체인의 커널 사이 간격이며 원인이 CPU라는 단정은 아니다.
- `speedup_vs_a16_same_waits`: 같은 layout/host-wait 정책의 A16 대비 chain wall 배율.

GPU 수치 검사를 먼저 실행하므로 새 chain의 `first_timed_post_validation`은 **cold JIT가 아니다.**
Cold compilation과 통상 반복 시간을 섞지 않는다. 기존 개별 tensor 비교기의 First 필드는
그 비교기에서의 최초 실행으로 별도 정의된다.

이 시간에는 **FA·norm·router 계산·residual·sampling·grammar·Vulkan handoff가 없다.**
따라서 전체 모델 TG를 예측하는 직접 측정값이 아니다. Multi-turn latency 문제의 원인이
이 MoE 체인 밖에 있으면 이 최적화로 해결되지 않을 수 있다.

### 정확성 기준

1. GPU에서 사용한 input A8 q/scale을 읽어 gate/up 기준을 만든다.
2. 캡처된 GPU gate/up에 CPU clipped SwiGLU를 적용해 epilogue 자체를 검사한다.
3. GPU hidden q/scale을 사용해 down의 패킹·누산·scale 적용을 검사한다.
4. GPU down 출력과 고정 route로 weighted sum을 독립 검사한다.
5. 원래 F32 입력 기반 전체 chain 대비 total NMSE를 별도로 기록한다.

각 단계의 arithmetic pass 기준은 finite, NMSE<=1e-6, max_abs/reference_RMS<=0.005다.
Quantizer는 GPU/CPU q·scale을 비교하고, RNE 경계 근처 ±1 차이만 따로 허용한다.
NaN/Inf는 clipping으로 숨기지 않고 먼저 status에 기록한다. 실패하면 속도 측정을 중단한다.

**`chain_total_nmse_vs_f32`는 통과 기준이 아니라 보고값이다.**
작업 품질·logits·speculative acceptance 허용 여부는 전체 모델과 실제 데이터로 평가해야 한다.
합성 zero/outlier 테스트를 모델 품질 인증으로 해석하지 않는다.

## API/통합 계약과 미구현 범위

`include/maple_moe.hpp`의 `enqueue_moe()`에 같은 SYCL context의 USM/input/workspace를 넘긴다.
같은 workspace를 동시에 사용하는 in-flight chain을 만들지 말고 dependency event 또는 별도
workspace를 사용한다. 예외 발생 시 queue를 drain하기 전에 버퍼를 해제하거나 fallback을
동일 output에 제출하면 안 된다. Status는 caller가 안전한 시점에 초기화하고 완료 후 검사한다.

사용자 log에서 통과한 A16/A8 DPAS 본체의 가중치 코드 mapping은 바꾸지 않았다.
추가한 것은 prequantized consumer, clipped SwiGLU/quant producer, routing-weight sum,
연결된 event chain, 비교기, hot-path device-query cache다.

아직 구현하지 않은 것은 Vulkan allocation import, 외부 semaphore 연결, llama scheduler의
MoE subgraph replacement, CPU fallback 정책, pure-SYCL Maple 전체 graph, large-prefill GEMM이다.
기존 v0.2의 `integration/ggml_decode_adapter.hpp`는 개별 Q1 node용 참고 adapter로 남겨두며,
새 full chain의 자동 scheduler 패치라고 취급하지 않는다.

## 여기서 수행한 테스트

```bash
cmake -S . -B build-cpu -DMAPLE_BUILD_SYCL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu
ctest --test-dir build-cpu --output-on-failure
```

CPU check: 기존 A16 2,096,458 + 기존 A8 1,888,002 + 새 MoE 97,428.
Python unit test 19개. GCC ASan/UBSan과 Clang CPU MoE 검사도 통과했다.
상세 로그는 `validation/`에 있으며 이전 버전 검증은 `validation-v0.1/`, `validation-v0.2/`다.
**새 GPU 수치/속도/PowerShell 검증은 로컬에서 위 명령으로 수행해야 한다.**
