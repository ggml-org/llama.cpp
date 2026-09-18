# Maple TQ2 → W2A16 XMX 실험 커널 v0.1

## 상태부터 명확히

**구현한 것:** Intel SYCL/ESIMD DPAS 커널, 기본 TQ2 직접 읽기, 같은 크기의 tile8 재배열,
F16/F32 activation 입력, 선택 expert별 matvec, 두 projection 동시 계산, split-K,
GPU 수치 검사/벤치마크, GGUF tensor 추출기, ggml Q=1 view adapter.

**실제로 확인한 것:** 이 패키지의 CPU 테스트와 GGUF 추출기 테스트. CPU 테스트는
TQ2 code 순서, VNNI 재배열, expert 주소, 입력 broadcast, split-K 수학을 검사한다.

**아직 확인하지 못한 것:** SYCL 컴파일, A750 실제 실행/ISA/속도, 모델 전체 품질,
현재 로컬 llama dispatcher/Win32 Vulkan↔Level Zero 동기화 연결.
이 환경에는 oneAPI SYCL compiler와 A750이 없었다. CPU 통과를 GPU 통과로 간주하지 않는다.

이 패키지는 `ggml-vulkan.dll`을 교체하지 않고, 기존 hybrid 설정도 바꾸지 않는다.
`integration/ggml_decode_adapter.hpp`는 실사용 가능한 shape/type 검사와 device view 연결을
제공하지만, **현재 로컬 dispatcher에 삽입된 패치는 아니다.** 서버 TG 향상은 아직 측정하지 않았다.

## 무엇이 XMX인가

`mul_mat_vec_tq2_0_f16_f32` 이름은 저장된 weight가 F16이라는 뜻이 아니다.
확인한 remote `fed6590f`에서는 TQ2 weight + F16 activation + F32 output이며,
전용 TQ2 matvec 안의 계산은 float FMA다. 같은 스냅샷의 MoE ID shader는 F32 activation
variant를 생성한다. `flash_attn_f32_f16_aligned`는 FA 이름이지 weight 연산 이름이 아니다.

본 구현은 다음을 호출한다.

```cpp
acc = sycl::ext::intel::esimd::xmx::dpas<8, 1, float>(acc, b_fp16_vnni, a_fp16);
```

DG2 기준 ExecSize=8, RepeatCount=1, K=16이다. weight 256개마다 원래 F16 scale을
읽어 {-d, 0, +d}를 register에서 만들고 FP16×FP16/FP32 누산한다.
F32 activation은 register에서 F16으로 반올림한다. F16 입력은 그대로 읽는다.
INT8 activation 양자화, native INT2 DPAS, FP16 weight 전체 펼치기는 하지 않는다.

컴파일러가 출력한 ISA의 FP16 DPAS 사용 여부는 로컬 GPU 도구로 별도 확인해야 한다.
벤치의 `requested_instruction` 필드는 **관측한 ISA가 아니라 요청한 소스 연산**이다.

## 설계/지원 범위

| 항목 | 구현 |
|---|---|
| Weight type | 표준 GGML TQ2_0 type=35, block 256, qs[64] + F16 d, 총 66 bytes |
| Native layout | 원래 TQ2 직접 읽기; 모델의 GPU 사본 추가 불필요 |
| Tile8 layout | 8 rows × 256 K를 528 bytes로 재배열; 정확히 같은 byte 수 |
| 원래 scale | 모든 256-block scale 보존; row 전체 같은 scale이라고 가정하지 않음 |
| Shape | K % 256 = 0, M % 8 = 0; Maple gate/up K2048/M512, down K512/M2048 |
| Expert | 각 job의 실제 expert ID로 선택; nonzero/마지막 expert 주소 포함 |
| Activation | F32→F16 register 변환 또는 기존 F16; gate/up broadcast와 down의 expert별 입력 |
| Output | F32 [tokens, topk, M]; router 가중치 합산은 하지 않음 |
| Small-Q | 벤치는 Q=1..8; 각 query를 독립 RepeatCount=1로 처리; prefill GEMM 아님 |
| Paired | 같은 shape의 gate/up 동시 계산, activation load 공유; 두 출력 유지 |
| Split-K | 1/2/4/8 중 K/256을 나누는 값. K512에서는 1/2만 가능 |
| Async API | dependency event 수신, completion event 반환; enqueue 내부 host wait 없음 |
| GPU 제한 | A750/A770 allowlist; 다른 Intel 세대를 같은 ISA 형상으로 실행하지 않음 |

Tile8은 가중치 값을 다시 양자화하지 않는다. 128개의 DPAS B-half를 다음 순서로 만든다.

```
B index = (k_in_16 / 2) * 16 + output_in_8 * 2 + k_in_16 % 2
```

TQ2의 reserved code 3(일반 dequant에서 +2*d)와 non-finite scale은 현재 pilot 대상이 아니다.
load 시 검사에서 거절해야 한다. F32 activation의 half overflow/NaN/Inf를 clamp하지 않는다.
벤치는 이를 거절하며, 서버 연결 시에도 명시적 precision policy/validation이 필요하다.

## Windows 빌드

oneAPI와 MSVC 환경이 초기화된 PowerShell에서 압축 해제한 디렉터리로 이동한다.
현재 서버를 내릴 필요는 없지만 성능 측정 시에는 다른 GPU 작업을 중지한다.

```powershell
.\scripts\build_windows.ps1
.\scripts\run_smoke.ps1 -Device A750
```

icx가 PATH에 없다면 설치된 oneAPI developer shell을 사용한다. 이 스크립트는
설치나 시스템 환경변수 변경을 하지 않는다. `cl.exe`로 SYCL 커널을 컴파일하지 않는다.
필요하면 `-Compiler "실제 icx.exe 경로"`를 전달한다.

빌드 결과:

```
build/maple-w2-bench.exe
build/maple-reference-tests.exe
build/sycl-build.log
build/smoke/*.log, *.csv
```

GPU smoke가 FAIL/컴파일 오류이면 해당 파일을 보존하고 기존 Vulkan을 유지한다.
이 패키지를 성공한 GPU 바이너리로 전제하지 않는다.

Linux oneAPI 환경에서는:

```bash
cmake -S . -B build-sycl -G Ninja -DMAPLE_BUILD_SYCL=ON -DCMAKE_CXX_COMPILER=icpx
cmake --build build-sycl
./build-sycl/maple-w2-bench --help
```

## 실제 Maple weight 비교

원본 GGUF를 변경하지 않고 한 expert tensor만 추출한다. GGUF library/torch/numpy는 필요 없다.
먼저 파일에 실제 존재하는 이름을 확인한다.

```powershell
python .\tools\extract_tq2.py --model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf" --list
```

출력에 아래 이름이 존재하는 경우:

```powershell
python .\tools\extract_tq2.py --model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf" `
  --tensor "blk.0.ffn_gate_exps.weight" --output .\gate.mw2
python .\tools\extract_tq2.py --model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf" `
  --tensor "blk.0.ffn_up_exps.weight" --output .\up.mw2
python .\tools\extract_tq2.py --model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf" `
  --tensor "blk.0.ffn_down_exps.weight" --output .\down.mw2

.\scripts\run_sweep.ps1 -Weights .\gate.mw2 -Weights2 .\up.mw2 -Tokens 1
.\scripts\run_sweep.ps1 -Weights .\down.mw2 -Tokens 1 -PerSelection 1
```

레이어 번호/이름은 `--list` 결과를 따른다. type42 custom Q2 파일은 거절한다.
.mw2와 provenance JSON에는 선택 tensor만 저장하며 원본 파일은 읽기 전용이다.

실제 decoder 입력을 재생하려면:

```powershell
.\scripts\run_sweep.ps1 -Weights .\gate.mw2 -Weights2 .\up.mw2 `
  -XFile .\gate-x.f32 -IdsFile .\gate-ids.i32 -Tokens 1 -InputType f16
```

`XFile`은 항상 little-endian F32 캡처 파일이다. `InputType=f16`은 이를 F16으로 바꿔
GPU에 올린다. 원래 F16 캡처는 먼저 같은 값의 F32로 저장한다.
X shape: gate/up [Q,K], down [Q,topk,K]. IDs: [Q,topk] int32.
파일을 생략하면 실제 weight + **합성 activation/합성 라우팅**이며 실사용 trace가 아니다.
`id-mode=rotate`는 host가 준비한 다른 expert ID를 사용해 같은 expert만 계속 cache-hit하는
측정을 피한다. 실제 model trace나 DRAM cold-cache 상태를 완전히 재현하는 것은 아니다.

## 수치 검증과 시간의 의미

비교값을 분리한다.

1. GPU vs 같은 activation 정밀도의 FP64 누적 gold: packing/주소/누산 오류 검사.
2. GPU vs 원래 F32 activation gold: F16 변환까지 포함한 총차이.
3. F16 activation gold vs F32 activation gold: activation 반올림 자체의 오차.

기본 GPU 산술 검사 기준: finite, NMSE <= 1e-6, max_abs/reference_RMS <= 0.005.
이 기준은 kernel 검사용이지 agent/model 품질 보증이 아니다. F16 activation에 따른
품질 허용 여부는 실모델 logits/작업 성공률로 따로 판단한다.

벤치가 측정하는 값:

```
main_us                 DPAS/FMA 커널 event 시간
reduce_us               split-K reducer event 시간
main+reduce             두 event 시간 합
 event_span_us          main 시작 → reduce/출력 완료 (같은 SYCL queue 구간)
 submit_wait_us         CPU enqueue 시작 → completion wait 완료
 first_*                첫 실행/JIT 포함값, 반복 실행과 분리
 median / p95           반복 측정값
```

**포함되지 않는 것:** 최초 weight upload/repack, output readback, Vulkan↔SYCL handoff,
FA, routing/softmax, SwiGLU, host sampler, grammar, 전체 request 시간.
`mode=fma`는 이 패키지의 단순 SYCL comparator이지 기존 Vulkan shader의 복제본이 아니다.
따라서 XMX/FMA 벤치 배율을 그대로 현재 llama TG 개선율로 쓰지 않는다.

## Hybrid 연결 시 반드시 지킬 계약

1. 로컬 최신 dispatcher에서 **plain Q=1 MUL_MAT_ID**에만 먼저 opt-in한다.
   adapter의 fallback reason이 있으면 기존 Vulkan을 유지한다. 기본 enable은 false다.
2. weights, activation, IDs, output은 **같은 SYCL context에 import된 실제 GPU allocation**이어야 한다.
   `ggml_tensor::data`를 임의로 SYCL USM pointer로 해석하지 않는다.
3. Vulkan producer 완료와 SYCL consumer 시작, SYCL 완료와 다음 Vulkan consumer의 가시성을
   확보한다. zero-copy는 zero-synchronization이 아니다. 연결 비용도 따로 측정한다.
4. `enqueue()`의 dependency/completion event와 기존 external-memory bridge를 연결한다.
   이 패키지는 Win32 handles, Vulkan semaphores, queue ownership transfer를 추측해 구현하지 않는다.
5. scratch/IDs/status를 node마다 malloc/free하지 않는다. 생존하는 graph/할당에 맞게 재사용하고,
   concurrent in-flight 호출은 별도 scratch를 쓴다.
6. 원래 dispatcher가 뒤따르는 bias/scale/route weighting/GLU를 fused 실행했다면,
   이를 건너뛴 채 plain kernel 결과를 완료 처리하지 않는다. adapter는 이런 경우 거절한다.
7. native layout부터 연결하면 GPU weight 재배열 사본이 필요 없다. tile8을 채택할 때는
   원본+재배열 두 GPU 사본을 상주시키지 말고 모델 allocator/소유권을 명시적으로 처리한다.
8. 동기적인 unsupported는 fallback 가능하지만 비동기 device fault를 무시하고 결과를
   정상 처리해서는 안 된다. fault 시 실행을 중단하고 진단 로그를 남긴다.
9. decoder GPU 시간과 wall time을 모두 비교한다. CPU grammar/FA/VRAM paging이 지배적이면
   weight kernel만 빨라져도 TG 변동은 남을 수 있다.

Adapter 사용의 핵심은 다음과 같다. imported 포인터/producer event는 기존 bridge가 제공해야 한다.

```cpp
maple_w2::DeviceProblem view;
const char *reason = maple_w2::make_ggml_decode_view(
    node, imported, maple_w2::Layout::native_tq2,
    weights_validated_at_load, allow_activation_rounding, has_fused_following_nodes, view);
if (reason) {
    // keep the current Vulkan dispatch
} else {
    // same context; persistent scratch; no host wait inside enqueue
    auto run = maple_w2::enqueue(existing_queue, view, options, persistent_scratch, producer_events);
    // connect run.done to the existing Vulkan handoff; do NOT just mark node done here
}
```

ggml adapter 자체는 현재 로컬 헤더/dispatcher와 compile 확인되지 않았다.
실서버 연결을 완료했다고 보고하지 않는다. 로컬 최신 소스가 remote 스냅샷과 달라
자동 patch로 동기화 규약을 바꾸지 않았다.

## 소스/근거

확인한 사용자 repo 스냅샷: `fed6590fd2e64da55baa4a31788dfb12454b1844`.
기존 로그의 로컬 build `a810870e6`와 동일하다고 가정하지 않았다.

- https://github.com/demilenos/llama.cpp/blob/fed6590fd2e64da55baa4a31788dfb12454b1844/ggml/src/ggml-common.h
- https://github.com/demilenos/llama.cpp/blob/fed6590fd2e64da55baa4a31788dfb12454b1844/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_tq2_0.comp
- https://github.com/demilenos/llama.cpp/blob/fed6590fd2e64da55baa4a31788dfb12454b1844/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp
- https://github.com/demilenos/llama.cpp/blob/fed6590fd2e64da55baa4a31788dfb12454b1844/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl
- https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/xmx/dpas.hpp
- https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/dpas.md
- https://arxiv.org/html/2508.06753v3 (저비트 kernel 연구; Xe2 결과를 A750 측정값으로 간주하지 않음)
