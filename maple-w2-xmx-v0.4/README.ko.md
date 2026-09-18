# Maple W2 XMX v0.4 — Vulkan architecture port v1

업로드한 `maple-w2-xmx-v0.4-source.zip`에 대한 **증분 소스 이식본**이다. 별도 로컬 수정본 복원이나 llama.cpp 기준선 재구성을 요구하지 않는다. 기존 v0.4의 static grouping·s2×s8 DPAS·G32/H32·SwiGLU·split-K를 보존하고, Vulkan 실험에서 필요성이 확인된 workload/검증 구조를 추가한다.

**실제 SYCL/ESIMD 컴파일, Windows 빌드, A750 실행·속도·전체 모델 품질은 아직 검증하지 않았다.** 이 환경에는 oneAPI SDK/GPU가 없다. CPU reference, 실제 신규 커널 본문의 scalar-DPAS 에뮬레이션, 실제 host enqueue DAG의 test-double 검사는 실행했다. `validation-port/`의 결과를 GPU 검증으로 해석하지 않는다.

## 무엇을 옮겼나

| 항목 | 구현 | 기본 동작 |
|---|---|---|
| Static grouping / 원래 slot 복원 | v0.4 구현 유지 | 기존 `expert_grouping` 설정 유지 |
| Workload dispatcher | `inherit_v04 / direct / grouped / auto_select` | `inherit_v04`; auto 미보정은 direct |
| Weight 재사용 | 같은 expert의 2·4 assignment가 B tile을 공유하는 새 ESIMD 커널 | gate/down 각각 T1, 신규 경로 OFF |
| Projection별 선택 | `gate_tokens_per_tile`, `down_tokens_per_tile` 독립 | 1 / 1 |
| Grouping–quant fork/join | quant는 입력 deps, grouping도 입력 deps, gate는 양쪽을 join | OFF; OOO에서만 overlap 가능 |
| A8 producer 공유 | 기존 quantizer를 `enqueue_a8_quant()`로 노출; fresh prequantized input 지원 | 기존 staged input |
| 수치 검증 | RNE contract 감사 + 실제 GPU q/scale로 stage-local oracle + raw dump/freeze-input | 원래 quantizer 수학 보존 |
| 계측 | producer DAG 검증, 전체 GPU span, interval union, overlap·idle 분리 | 새 비교기에서 사용 |

**T2/T4는 DPAS RepeatCount=2/4 구현이 아니다.** 기존 `dpas<8,1,...s2,s8>`를 각 token row에 대해 수행하고 B load와 scale load를 공유한다. DG2 출력 채널 8개와 token 8개를 혼동하지 않는다. T2/T4를 자동 최적값으로 간주하지 않는다.

## 가장 빠른 시작 — Windows

압축을 별도 디렉터리에 푼 뒤:

```powershell
.\build_architecture_and_test.cmd -Device A750 -Suite smoke
```

기존 oneAPI/MSVC 초기화 경로를 사용한다. 새 라이브러리·비교기를 빌드하고 작은 synthetic 2-case 검사부터 수행한다. 기존 llama-server DLL, 모델, 환경변수 설정은 바꾸지 않는다. oneAPI가 이미 초기화되어 있으면:

```powershell
.\scripts\build_architecture_and_test.ps1 -Device A750 -Suite smoke
```

GPU 검사를 실행하지 않고 빌드만:

```powershell
.\build_architecture_and_test.cmd -BuildOnly
```

빌드 결과는 `build/maple-architecture-compare.exe`이다. 그 후 재컴파일 없이:

```powershell
python tools/run_architecture_suite.py `
  --exe build/maple-architecture-compare.exe `
  --device A750 --suite quick --out build/arch-quick --dump-contract

python tools/analyze_architecture_results.py build/arch-quick
```

`quick`: Q={1,13,184,512} × 6개 recipe = 24 cases. `full`: Q={1,2,4,8,16,32,64,128,184,256,512,1024,2048} × 6 = 78 cases. 각 case 안에서 같은 binary·입력·ID frame으로 baseline/candidate AB/BA를 비교한다. 모든 측정에 그 call의 grouping/descriptor 비용이 들어간다.

기본 weight/activation은 synthetic이다. 기존 추출기를 통해 얻은 **동일 layer의** real capsule 세 개를 넣으면 weight만 실물이 된다:

```powershell
python tools/run_architecture_suite.py `
  --exe build/maple-architecture-compare.exe --device A750 `
  --suite quick --out build/arch-real `
  --gate C:\AI\capsules\gate.mw2 `
  --up C:\AI\capsules\up.mw2 `
  --down C:\AI\capsules\down.mw2
```

실제 activation/router를 쓰려면 개별 비교기의 `--x-file`, `--ids-file`, `--routes-file`을 사용한다. real weights + synthetic activation을 전체 모델 검증이라고 부르지 않는다.

## 개별 A/B

```powershell
.\build\maple-architecture-compare.exe `
  --device A750 --tokens 184 --k 2048 --hidden 512 --experts 256 --topk 8 `
  --candidate grouped --gate-tile 2 --down-tile 1 --overlap 1 --in-order 0 `
  --gate-split 1 --down-split 1 --local 4 `
  --warmup 3 --repeats 28 --dump-contract 1 --out build/q184-t2-1
```

| 스위치 | 의미 |
|---|---|
| `--candidate direct/grouped/auto` | 후보 schedule; baseline은 direct 고정 |
| `--gate-tile 1/2/4` | gate/up의 token tile |
| `--down-tile 1/2/4` | down의 token tile |
| `--overlap 0/1` | grouping/quant dependency를 분리 |
| `--in-order 0/1` | 1이면 queue가 직렬화; overlap 설정만으로 병렬화되지 않음 |
| `--auto-min-tokens N` | N=0은 미보정/direct, N>0은 명시적 임계점 |
| `--auto-min-mean M` | `tokens*topk / experts`의 평균 assignment 하한; 실제 skew 통계 아님 |
| `--dump-contract 1` | 두 snapshot의 raw q/scale·중간값 보존 |
| `--freeze-input 1` | CPU RNE q/scale을 두 arm에 공급하는 원인분리 검사; 성능 선정에서 제외 |

Auto는 N=1을 direct로 둔다. forced grouped는 N=1 비용 비교를 허용한다. 미보정 auto에 Vulkan의 N=184 결과를 XMX 임계점으로 몰래 적용하지 않는다. 입력·hidden group은 여기서 G32/H32 고정이다. tile 1/2/4와 양자화 group 32/128은 별개의 변수다.

## Q8 계약: Vulkan 값을 복사하지 않은 이유

Vulkan 실험의 첫 code `111 ↔ 110`은 half-away 반올림 앞의 division/reciprocal 차이였다. **XMX v0.4는 RNE(ties-to-even)**이므로 같은 110.5는 원래부터 110이다. 생산 quantizer를 half-away로 바꾸거나 strict division을 강제하지 않았다.

`audit_a8()`는 다음을 별도로 기록한다.

* CPU division oracle 대비 code·scale 차이, scale ULP.
* 실제 GPU scale로 설명되는 code 차이와 normalized half-boundary 근처 차이.
* 진짜 잘못된 code/scale, -128, nonfinite.

기본 scale relative bound=5e-6, normalized error window=2e-4는 명시적 **검증 정책**이다. 모든 ±1 code를 통과시키지 않는다. 실제 DPAS projection은 **GPU가 만든 q·scale을 동일하게 사용한 CPU reference**와 비교하며, 양자화기 자체의 독립 검사도 남긴다. 전체 모델의 양자화 손실을 허용했다고 해석하지 않는다.

```powershell
python tools/audit_quant_dump.py build/q184-t2-1/snapshot-0-candidate
```

Raw binary는 little-endian이다. `numerical.csv`는 최대 4개 token(0,12,중간,마지막)의 모든 expert 출력에 대한 stage oracle이고, baseline/candidate 전체 tensor·quant plane 비교는 전 token이다. 샘플 CPU 검사와 full GPU pair 비교를 구분한다.

## 지연시간 해석

`wall_us`는 enqueue 진입부터 최종 wait 반환까지이며 upload/readback은 밖이다. `gpu_span_us`는 모든 기록 stage의 min(start)~max(end)이다. fork/join에서는 kernel 시간 합이 span보다 클 수 있으므로 다음을 분리했다:

```
kernel_sum = 각 stage duration 합
union_busy = stage 실행구간들의 합집합 길이
overlap    = kernel_sum - union_busy
gpu_gap    = gpu_span - union_busy
```

`union_busy`는 profiler interval의 합집합이지 GPU의 모든 EU가 바빴다는 뜻은 아니다. `core_gpu_span`도 순수 matmul 시간이 아니라 최초 non-group stage~최종 stage이다. DAG 부모 이벤트가 끝난 뒤 자식이 시작했는지 검사한다. submission order로 직렬 실행을 강요하지 않는다.

성능 분석기는 같은 case의 paired wall만 비교하고, freeze-input 결과와 단일 sample 결과를 production 권고에서 제외한다. universal auto threshold나 전체 LLM t/s를 추정하지 않는다.

## 기존 llama.cpp에 반영

`docs/INTEGRATION_PORT.ko.md`를 따른다. 제공한 module 기준 patch와 hash-checked overlay를 포함한다. 현재 llama.cpp 전체를 reset/clean/rebase할 필요 없다.

```powershell
# 기본: 변경하지 않고 정확한 v0.4 파일과 맞는지만 확인
python tools/apply_architecture_port.py --target C:\AI\llama-src\YOUR_W2_MODULE

# 전체 preflight 통과 후 파일별 backup을 남기고 적용
python tools/apply_architecture_port.py --target C:\AI\llama-src\YOUR_W2_MODULE --apply
```

`YOUR_W2_MODULE`은 실제 `include/maple_moe.hpp`와 `src/maple_moe.cpp`가 있는 module root다. 파일을 ggml 소스에 직접 합쳐 구조가 달라졌다면 overlay를 강제로 쓰지 말고 unified diff의 대응 부분을 적용한다. 알려지지 않은 llama.cpp source path를 추정해 덮어쓰지 않는다.

헤더/구조체가 확장됐으므로 관련 object/library를 모두 재빌드한다. 구버전 객체와 신버전 헤더를 혼합하지 않는다. 프로파일링과 raw dump는 비교기에서만 수행하며 생산 enqueue에는 강제 host wait/readback이 없다.

## 범위 밖

이 port는 기존 XMX 모듈의 MoE 범위를 개선한다. full-router/QKV의 ggml graph replacement, Vulkan↔SYCL memory import·queue handoff, persistent work-stealing scheduler, DPAS RepeatCount>1 GEMM, 실제 모델 perplexity/code-task 평가까지 구현했다고 주장하지 않는다. 특히 다른 runtime의 VkBuffer나 raw device 주소를 USM으로 직접 넘기지 않는다.

기존 README는 `README.v0.4.ko.md`에 보존했다. 원본 `FILE_SHA256SUMS.txt`/`SOURCE_PROVENANCE.json`는 v0.4의 과거 기록이다. 이 port의 현재 identity는 `PORT_SHA256SUMS.txt`, `PORT_PROVENANCE.json`이다.
