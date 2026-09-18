# 기존 llama.cpp XMX v0.4 module에 적용하는 계약

## 1. 변경 경계

현재 llama.cpp에 들어 있는 XMX tree를 기준으로 삼는다. standalone 실험기를 새 production tree로 만들거나 기존 hybrid/SWA를 버리지 않는다. 이 패키지는 업로드한 v0.4 module의 증분 변경이며, 사용자의 전체 llama.cpp checkout을 열어 수정한 결과는 아니다.

* 기존 `maple_w2a8.cpp`의 s2 packing·DPAS·scale accumulation·quantizer 수학은 그대로.
* 기존 `maple_moe.cpp`의 SwiGLU/hidden quant kernel 수학은 그대로.
* `expert_grouping.cpp`의 count/prefix/stable scatter는 그대로이며 token descriptor builder만 추가.
* 새 `src/maple_w2a8_grouped.cpp`를 기존 XMX library source 목록에 추가.
* 헤더가 늘어나므로 해당 라이브러리와 모든 caller를 함께 재빌드.

`v0.4-to-vulkan-port-v1.patch`는 module root 기준 unified diff다. `git apply --check` 뒤 적용할 수 있다. 원본 payload와 대상 파일의 SHA를 검사하는 `tools/apply_architecture_port.py`는 unrelated llama.cpp 파일을 건드리지 않는다. 파일 구조가 변경된 통합본에는 강제 적용하지 않는다.

## 2. 기본 경로는 v0.4와 호환

```cpp
MoeOptions o;                // schedule = inherit_v04
// 기존 코드의 o.expert_grouping=false/true가 계속 적용된다.
auto run = enqueue_moe(q, problem, o, workspace, upstream_events);
// 생산 경로에서는 여기서 wait하지 말고 downstream에 run.done 전달.
```

기존 struct 멤버 뒤에 옵션/workspace를 추가했다. source-level aggregate initialization은 유지하지만 binary ABI 보장은 없다. 기존 `grouping_compare`는 그대로 남아 serialized grouping baseline을 비교한다.

## 3. 실험용 후보

```cpp
MoeOptions o;
o.path = MoePath::a8_gluquant;
o.input_group = o.hidden_group = 32;
o.schedule = MoeSchedule::grouped;
o.overlap_grouping_quant = true;
o.gate_tokens_per_tile = 2;
o.down_tokens_per_tile = 1;
```

Group reuse는 G32/H32, s2tile8, fresh prequantized q/scale에 한정된다. native TQ2는 기존 model-load의 s2tile8 변환을 이용하고, 전체 INT8/FP16 weight shadow를 추가하지 않는다. 양자화된 weight code 3을 +2로 처리하는 일반 Vulkan semantics를 복사하면 안 된다. v0.4의 strict ternary 검증을 유지한다.

## 4. Persistent workspace

기존 grouping buffer 외에 T>1일 때만 다음 descriptor planes를 확보한다.

```cpp
const size_t jobs = size_t(problem.tokens) * problem.topk;
const size_t capacity = expert_token_tile_capacity(jobs, problem.gate_shape.experts, T);
// 각각 capacity개의 int32: first, length, expert
// 한 개 int32: total
ExpertTokenTileView tv{first, length, expert, total, capacity, T};
```

capacity 상한은 `min(jobs, floor((jobs+(E+1)*(T-1))/T))`이며 invalid-ID bucket까지 포함한다. 실제 tile 수는 GPU가 `total`에 기록하고 kernel이 읽는다. host readback/indirect launch가 필요 없다. 상한보다 적은 tile은 early return한다.

Gate와 down의 T가 같으면 gate descriptor를 공유한다. 다르면 별도 descriptor buffer가 필요하다. 같은 workspace를 동시에 실행하는 request들이 덮어쓰면 안 된다. grow/reuse는 기존 allocator에서 완료 event를 존중한다.

여기서 tile은 (expert, sorted assignment 시작, 유효 길이)다. `first`는 원래 token 번호가 아니다. 실행 중에는 `sorted_to_job`을 거쳐 원래 job을 찾는다. input row는 `job/topk`, hidden row는 `job`, output은 원래 `job*M+channel`이다. 최종 route weighted sum은 기존 top-k slot 순서를 유지한다.

## 5. Fork/join과 외부 producer

```
upstream_events ── grouping count/prefix/scatter ── token descriptors ─┐
          └─────── input quant ──────────────────────────────────────┤
                                                                    gate/up
                                                                       ↓
                                                               SwiGLU + hidden quant
                                                                       ↓
                                                                      down
                                                                       ↓
                                                               weighted sum → done
```

Grouping과 quant는 서로 쓰는 buffer가 다르다. `overlap_grouping_quant=true`일 때 input quant는 grouping 완료를 기다리지 않지만 gate는 두 producer 모두 기다린다. in-order queue는 여전히 직렬화한다. out-of-order queue도 실제 overlap을 보장하지 않으므로 stage timestamp로 확인한다.

공통 입력 quant를 다른 projection과 공유할 때:

```cpp
auto input_ready = enqueue_a8_quant(q, problem.x, ActivationType::f32,
    problem.tokens, problem.gate_shape.k, 32, workspace.input_a8, upstream_events);
o.input_prequantized = true;
auto run = enqueue_moe(q, problem, o, workspace, {input_ready});
```

입력 q/scale/invalid가 **이번 call**의 데이터여야 한다. stale activation cache 기능이 아니다. external memory import 및 cross-runtime dependency는 기존 bridge가 보장해야 한다. 이 API는 raw Vulkan 주소를 자동 변환하지 않는다.

## 6. Auto와 per-projection tuning

Auto는 `min_tokens=0`이면 direct만 사용한다. XMX 실측을 확보한 뒤 해당 device/build/shape에 적용할 하한을 명시한다. 평균 assignment는 Q*topk/E 추정이며 skew를 알지 못한다. forced grouped를 남겨 crossover 근처의 direct와 항상 A/B할 수 있게 했다.

T2/T4에서는 B tile을 여러 token에 재사용하지만 각 token의 연산은 RepeatCount=1이다. 기존 N8은 출력 채널 8개다. 향후 multi-row DPAS는 operand/accumulator layout과 별도의 ISA probe가 필요한 다른 패치다.

Gate와 down의 T를 독립적으로 선택한다. QKV reuse와 MoE reuse의 Vulkan 결과를 하나의 전역 스위치로 묶지 않는다. 이 module에는 QKV/router graph ownership이 없으므로 둘을 임의로 추가하지 않았다.

## 7. 오류 및 회귀 처리

* invalid capacity/configuration은 첫 제출 전 거부한다.
* invalid expert는 기존 status error를 남기며 정상 expert로 바꾸지 않는다.
* submission 중 예외가 나면 queue를 drain한 뒤 workspace를 재사용한다. 실패 도중 fallback이 같은 출력에 쓰게 하지 않는다.
* CPU/GPU 산술 차이는 quant contract와 projection 검사를 분리해 본다. checker tolerance를 전체적으로 늘리지 않는다.
* 공개 run.done으로 모든 내부 producer가 transitive하게 연결된다. mocked host DAG 검사는 이를 검증하지만 실제 GPU runtime 검증의 대체물은 아니다.
