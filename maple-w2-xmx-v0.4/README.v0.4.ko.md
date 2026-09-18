# Maple W2 XMX optimize v0.4 — per-expert token grouping 한 가지 변경

## 상태

이 패키지는 **소스와 Windows 빌드/자동 비교 도구**다. 작성 환경에서 CPU reference,
회귀 검사와 sanitizer를 실행했다. **신규 SYCL 컴파일, A750 GPU 실행, 성능 결과는 아직 없다.**
GPU PASS나 속도 향상을 미리 주장하지 않는다. 기존 llama DLL, 서버 설정, GGUF를 수정하지 않는다.

제공된 `maple-w2-xmx-v0.3-source.zip`을 기반으로 했다. 사용자의 Q-sweep v2 ZIP에는
측정 자료만 있고 Q 확장 소스 diff/바이너리가 없었다. 따라서 v0.3의 Q<=8 guard를 양쪽에
동일하게 Q<=2048로 넓혔다. **과거 수치와의 차이보다 같은 v0.4 바이너리의 두 경로를 비교한다.**
기존 명령의 상세 변경은 `v0.3-to-v0.4.patch`, 파일 출처는 `SOURCE_PROVENANCE.json`에 있다.

## 이번 빌드에서 바꾼 것 / 바꾸지 않은 것

기준선은 다음으로 고정한다.

```text
W2A8 / native signed INT2 x INT8 DPAS
s2tile8 / G32 / H32 / gluquant / async
Gate split=1 / down split=1 / local=4
DPAS RepeatCount=1 / 출력 채널 8개 tile
```

비교 경로는 정확히 두 가지다.

1. **baseline**: 기존 token-major `(token, selected-expert-slot)` job 순서.
2. **expert-grouped**: `(expert ID, original job index)`로 안정 정렬한 실행 순서.

이번에는 **multi-token DPAS / grouped GEMM, 출력 tile 확대, split 최적화,
H128, queue graph replay, Vulkan interop를 넣지 않았다.** 정렬과 GEMM을 동시에 바꿔
원인을 다시 분리하는 상황을 피한다. v0.4가 측정하는 것은 **expert-major job 배치로
얻는 지역성의 효과 - 매 호출 grouping 비용**이다. GPU가 workgroup을 반드시 순번대로
실행한다고 보장하는 설계는 아니다.

A8의 unpack/DPAS/scale 누산 본체와 clipped SwiGLU 함수는 v0.3과 동일하다.
`evidence/v03_*_excerpt.txt`와 Python 회귀 검사가 해당 구간을 문자 단위로 비교한다.
A16은 라이브러리에 보존했지만 기본 v0.4 성능 스윕에는 넣지 않았다.

## 데이터 흐름

```text
매 호출의 ids [Q,topk]
   ↓ GPU group_count
   ↓ GPU group_prefix
   ↓ GPU group_scatter       ← stable sorted_to_job, 매번 재계산
input_quant                  ← 기존 G32
   ↓
gate/up: grouped index → original job → 기존 DPAS
   ↓ 원래 token/selection 위치에 저장
SwiGLU + hidden quant        ← 기존 H32, 기존 순서
   ↓
down: 같은 sorted_to_job 공유 → 기존 DPAS
   ↓ 원래 token/selection 위치에 저장
weighted_sum                ← 원래 slot 순서대로 합산
```

**가중치, activation, hidden 전체를 정렬 복사하지 않는다.** 인덱스만 재배열하고
원래 주소를 읽고 쓴다. 따라서 추가 gather/scatter 데이터 커널이나 route 재정규화가 없다.
여기서 `group_scatter`는 작은 job-index 배열을 쓰는 단계다.

Grouping은 순수 SYCL GPU 3개 커널이고 event dependency로 직렬 연결한다.
입력 ID의 host readback, 매 enqueue 할당, 이전 토큰의 plan 캐시, 명시적 host wait는 없다.
입력 quant와 grouping을 겹치는 별도 스케줄링 최적화도 이번에는 하지 않았다.

Grouping workspace:

```text
counts        : int32 [experts+1]
offsets       : int32 [experts+2]
sorted_to_job : int32 [Q*topk]
```

마지막 bucket은 잘못된 expert ID를 위한 것이다. 잘못된 ID를 정상 expert에 섞지 않고,
기존 matvec status 검사가 실패하게 한다. Q2048/top8/experts256에서는 추가 workspace가
**67,596bytes (약 66.01KiB)**다. 새 weight 사본은 만들지 않는다.

구현은 per-expert workgroup의 scan 기반 안정 counting sort다.
128개 lane을 사용하며 ID 비교 작업은 대략 O(experts * jobs)다. sorting 자체가 공짜이거나
최적인 구현이라고 가정하지 않는다. 세 커널의 비용을 항상 측정한다. scan 사용 시 동일
workgroup의 모든 lane이 같은 collective 호출을 수행하도록 했다.
참조 API: https://github.khronos.org/SYCL_Reference/iface/group-algorithms-library.html

## 한 번에 빌드하고 Q 스윕

별도 `maple-w2-xmx-v0.4` 폴더에 풀고, 그 폴더에서 실행한다.

```powershell
.\build_and_test.cmd -Suite full -Device A750 -Model "C:\AI\models\maple-preview-TQ2_0-head-Q4_K.gguf"
```

기존에 추출한 동일 layer의 capsule을 사용하면 재추출하지 않는다.

```powershell
.\build_and_test.cmd -Suite full -Device A750 -CapsuleDir "C:\AI\llama-src\maple-w2-xmx-v0.3\build\results\YOUR-RUN\moe\weights"
```

`CapsuleDir`에는 `gate.mw2 / up.mw2 / down.mw2`가 필요하다. 서로 같은 layer에서 추출한
파일이어야 한다. `-Model`과 함께 사용하지 않는다. `-Python`으로 기존 Python venv를
지정할 수 있고, 기본 `-Layer`는 0이다. Python 3.10+와 oneAPI/MSVC 환경이 필요하다.
`.cmd`의 환경 초기화는 자식 프로세스에 한정한다. 전역 설치나 영구 환경 설정을 바꾸지 않는다.

| Suite | 실행 |
|---|---|
| smoke | Q1 작은 chain + Q13 out-of-order chain, 합성 가중치 |
| quick | smoke + 실제 모델 shape Q1/4/64/256/2048 |
| full | smoke + Q1/2/4/8/16/32/64/128/256/512/1024/2048 |

Full은 **14 cases, 28 variant summary**다. Model shape는 28 repeats, smoke는 최대 4 repeats다.
각 case에서 baseline과 grouped만 AB/BA 순서를 번갈아 측정한다. 그룹 크기와 split을
추가로 스윕하지 않는다. Q1에서도 자동으로 grouping을 끄지 않으므로 overhead도 드러난다.
기존 `-MoeOnly`를 붙여도 되지만 v0.4는 원래 grouping 비교만 수행한다.

다시 측정할 때:

```powershell
.\build_and_test.cmd -NoBuild -Suite full -Device A750 -CapsuleDir "C:\path\to\weights"
```

`-NoBuild`는 source와 exe hash를 확인한다. 소스가 변경됐거나 다른 바이너리면 재빌드를 요구한다.
이전 v0.3의 많은 variant를 다시 돌리고 싶을 때만 `build_and_test_v0.3.cmd`를 사용한다.

## 측정 범위와 결과 파일

```text
build/results/<timestamp>-v04-grouping-full/
  run.log
  grouping/
    probe.log
    grouping-build 정보는 suite_status.json의 build 항목
    all_grouping_summary.csv
    q_speedup.csv             ← 먼저 볼 파일
    suite_status.json         ← 소스·바이너리·가중치 hash, 실패 보존
    q0001 ... q2048/
      summary.csv
      samples.csv
      stages.csv
      routing_histogram.csv
      mapping_validation.csv
      pair_correctness.csv
      numerical.csv
      manifest.txt
```

| 지표 | 정의 |
|---|---|
| `wall_median_us`, `wall_p95_us` | enqueue 시작부터 최종 MoE completion까지. grouping 포함 |
| `group_kernel_median_us` | count+prefix+scatter의 GPU 실행시간 합. inter-kernel gap은 별도 |
| `core_gpu_span_median_us` | input quant 시작부터 expert 합산 완료까지 |
| `gate_up_median_us`, `down_median_us` | 가중치 행렬 연산 구간 |
| `gpu_span_median_us` | grouped이면 count부터, baseline이면 input_quant부터 최종 출력까지 |
| `gpu_gap_median_us` | 같은 반복의 GPU span - kernel sum |
| `speedup_vs_ungrouped_same_build` | 같은 바이너리 baseline wall 중앙값 / 해당 경로 wall 중앙값 |
| `paired_saved_median_us` | 같은 ID frame을 처리한 반복별 baseline-grouped 시간차의 중앙값 |
| `grouped_wins` | paired repeat에서 grouped가 빠른 횟수 |

**grouping을 미리 해 놓고 dot만 재는 경로는 없다.** 매 호출 새로 plan을 만든다.
Warmup/first는 repeat 통계에서 제외한다. first도 CPU/GPU 수치 검사 뒤라 cold JIT가 아니다.
같은 round에는 같은 입력과 같은 ID frame을 사용한다. 다음 round의 expert ID는 보통 달라진다.
고정 ID 파일을 줘도 grouping을 생략하지 않는다.

weight upload/repack, workspace allocation, 기준값 계산, 검사용 readback은 steady-state 시간 밖이다.
FA, norm, router 계산, residual, sampler, Vulkan handoff도 없다. **전체 모델 TG가 아니다.**
`-ScrubMiB` 기본값은 이전 테스트와 같은 0이다. 한 layer 반복에 따른 cache 영향을 없앤 결과라고
주장하지 않는다. 실제 weight를 지정해도 activation·IDs·routing weight는 기본 합성이다.

각 지표의 중앙값은 독립적으로 집계한다. median(kernel)+median(gap)이 median(span)과
정확히 같을 필요는 없다. 원인 분석은 samples/stages의 같은 반복 단위로 한다.
순이익을 판정할 때는 `core_gpu_span`의 개선만 보지 말고 **wall time**을 사용한다.

## 정확성 검사

GPU 실행 시:

1. 기존 signed INT2/INT8 DPAS probe.
2. 신규 grouping probe: invalid ID, 빈 expert, 모두 같은 expert, 128 경계 tail, 16384 jobs,
   같은 workspace에서 변경된 IDs, out-of-order dependency, 뒤쪽 canary.
3. 본 비교의 두 ID snapshot에서 **전체 counts/offsets/permutation**을 CPU reference와 비교.
4. 전체 Q의 gate/up/hidden/down/y를 baseline-grouped 간 비교. 출력은 검사 전 NaN으로 poison.
5. Input/hidden A8 code와 scale 배열은 baseline/grouped가 **bitwise 같아야** 통과.
6. CPU 산술 기준은 snapshot별 **최대 3개 token의 모든 selected expert/output**을 검사.
   큰 Q에서도 검증 시간이 폭증하지 않게 표본 CPU 검사와 전체 GPU A/B 검사를 구분했다.
7. 마지막 timed call의 grouping도 CPU reference와 비교하여 untimed plan 재사용을 배제.

출력의 full GPU A/B 허용치는 NMSE<=1e-10, max_abs/reference_RMS<=1e-4이며,
bitwise 일치 여부도 별도 기록한다. CPU stage 허용치는 기존 NMSE<=1e-6,
max_abs/reference_RMS<=0.005다. 실패하면 timing/집계를 중단하고 로그를 보존한다.

`chain_sample_nmse_vs_f32`는 최대 3개 token 표본의 기존 F32 체인 대비 보고값이다.
**전체 Q의 CPU gold 결과나 전체 모델 품질 인증으로 취급하지 않는다.**
그룹화 자체는 모델 연산·양자화 규칙을 바꾸지 않는다.

## 실제 activation/ID로 한 번 비교

```powershell
.\build\maple-grouping-compare.exe --device A750 --gate .\gate.mw2 --up .\up.mw2 --down .\down.mw2 --tokens 64 --topk 8 --x-file .\x.f32 --ids-file .\ids.i32 --routes-file .\routes.f32 --gate-split 1 --down-split 1 --local 4 --out .\build\results\captured-q64
```

Little-endian raw: x=[Q,K] F32, ids=[Q,topk] I32, routes=[Q,topk] F32.
입력과 모든 capsule은 같은 layer에서 캡처해야 한다. Routes는 이미 정규화되어 있어야 한다.
이 명령도 두 경로를 자동 비교한다. 새로운 Q마다 따로 코드를 수정할 필요는 없다.

## API / 수명 계약

`MoeOptions::expert_grouping=true`, `MoeWorkspace::grouping`을 설정한다.
모든 USM 포인터는 같은 context에서 접근 가능해야 한다. 동일 workspace를 동시에
in-flight인 두 호출에 쓰지 않는다. Producer dependencies를 정확히 넘긴다.
예외 후 queue를 drain하기 전에 버퍼를 해제하거나 같은 출력에 fallback을 쓰면 안 된다.

Grouping은 현재 **s2tile8/G32/H32/gluquant**에만 열려 있다. 다른 조합에 조용히 적용하지 않고
명시적으로 거절한다. 기존 baseline은 `expert_grouping=false`로 그대로 사용할 수 있다.

## Linux / CMake

CPU 검사만:

```bash
cmake -S . -B build-cpu -DMAPLE_BUILD_SYCL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

oneAPI가 있는 환경의 SYCL 빌드 (여기서는 미실행):

```bash
cmake -S . -B build -DMAPLE_BUILD_SYCL=ON -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build --target maple-grouping-compare -j
python tools/stamp_grouping_build.py --exe build/maple-grouping-compare
python tools/run_grouping_suite.py --exe build/maple-grouping-compare --out build/results/grouping-full --suite full --capsule-dir /path/to/capsules
```
