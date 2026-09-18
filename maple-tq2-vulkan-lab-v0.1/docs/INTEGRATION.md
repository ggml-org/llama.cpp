# Dirty llama.cpp 연결 계약 — 아직 적용하지 않은 작업

## 적용 전제

이 문서는 hook의 요구사항을 정의한다. 존재가 확인되지 않은 ggml API에 가짜 호출을 붙인 patch가 아니다. 원본 `ggml-vulkan.cpp`, graph builder, tensor loader, shader generator/CMake가 필요하다. 첨부 build는 `da0b59a62-dirty`이므로 해당 commit의 upstream 파일만 가져와 local hybrid 변경을 덮어쓰면 안 된다.

## Graph 경계

1. Maple graph의 router weight가 F32인지 확인한다. GEMV 결과를 기존 top-k의 정확한 tie/order·softmax 규약과 비교한다.
2. 동일 `input`, 동일 expert `ids`, 동일 device, 호환 shape/type/stride인 up/gate를 한 fused node 또는 backend fusion pattern으로 묶는다. 다른 consumer가 원래 up/gate 결과를 읽는 경우 삭제하면 안 된다.
3. Native TQ2 weights 세트와 bias 없는 projection만 첫 경로에 포함한다. LoRA, scale modifier, adapter, 다른 tensor type, 비연속 tensor, 다른 activation 종류에는 기존 경로를 유지한다.
4. Fused output은 clamped SwiGLU 뒤 `[assignment, FFN]`이다. down 입력의 expert 축을 `assignment/topk`로 잘못 축약하지 않는다.
5. Q/K/V는 virtual row concat을 통해 기존 buffers를 읽는다. QKV output의 view는 token stride가 fused row width임을 반영해야 한다. 단순히 각 projection을 contiguous라고 표시하면 두 번째 token부터 잘못 읽는다.
6. QKV fusion은 projection까지만 바꾼다. SWA의 RoPE, Global의 NoPE, norm, QK scale, KV cache update는 원래 graph 순서를 보존한다.

## Shader binding 및 파라미터

`project.comp` 공통 9개 binding:

| binding | 내용 |
|---|---|
| 0 | W0: single/up/Q TQ2 bytes |
| 1 | W1: gate/K TQ2 bytes |
| 2 | W2: V TQ2 bytes |
| 3 | activation: F32 bits 또는 Q8g32 packed INT8 |
| 4 | Q8g32 FP32 scales, F32에서는 dummy |
| 5 | direct: expert IDs / bucket: sorted assignment indices |
| 6 | bucket job array `uvec4(expert, begin, count, row_start)` |
| 7 | control `uint cursor, njobs, errors, reserved` |
| 8 | F32 output |

48-byte push constant:

```c
uint32_t k, m, n, topk;
uint32_t experts, schedule, per_assignment, nq;
uint32_t nk, o0, o1, o2;
```

`schedule=0`: dense implicit jobs. `1`: direct routed jobs. `2`: persistent bucket jobs. `3`: static bucket jobs.
`per_assignment=0`: up/gate의 input row=`assignment/topk`; `1`: down의 input row=`assignment`.
`o0/o1/o2`: 정렬된 descriptor 기준 짝수 byte offset. Descriptor offset 자체는 device의 `minStorageBufferOffsetAlignment`를 만족해야 한다. 각 weight 접근의 descriptor range에 4-byte padding을 포함한다.

## Scratch ownership 및 synchronization

요청/stream/graph replay 간에 하나의 global counter를 공유하지 않는다. 각 동시 실행 컨텍스트가 자기 `control/counts/offsets/cursors/sorted/jobs/Q8` scratch를 가져야 한다. Buffer를 다음 요청에 재사용하려면 해당 command completion을 확인한다.

순서:

```text
router projection → fused selection
→ clear counts/control → count → prefix → scatter
→ clear cursor/njobs only → build jobs → projection
→ optional Q8 quantize intermediate
→ clear cursor/njobs only → build down jobs → down
→ ordered expert reduction
```

실제 plan에서는 최초 control 초기화를 router 앞에 한다. Router가 남긴 오류를 중간에 지우면 안 된다. `build_jobs`는 errors를 보존하고 cursor/njobs만 reset한다. `njobs`가 capacity를 초과하면 errors bit를 켜며 projection은 실행하지 않는다.

각 화살표에서 shader-write→shader-read 및 필요 시 transfer-write→shader-read 의존성을 명시한다. 기존 backend command buffer 안에서 녹화하고 새 queue/device로 복사하지 않는다. 전체 host-side readback으로 `njobs`를 가져오지 않는다. Persistent kernel은 finite job cursor만 쓰며 다른 WG가 살아나기를 기다리는 inter-WG spin barrier는 없다.

## Precision/fallback

기본은 F32 activation 경로. Q8는 별도 opt-in으로 모델 수준 KLD/retrieval/code regression을 통과해야 한다. Q8g32 buffer를 기존 Q8_K/Q8_1 API에 넘기지 않는다. BF16/F16 reference와 동일 rounding이 필요하면 graph 경계의 cast를 명시적으로 재현한다. 현재 standalone code는 이를 자동 추측하지 않는다.

Router tie와 reduction 차이 때문에 expert가 바뀌면 먼저 동일 logits를 고정해 selection만 비교한다. 다음 단계에서 projection의 수치 차이를 검사한다. 코드 포맷 TQ2의 `3`은 표현상 +2이므로 decoder에서 임의로 0 처리하지 않는다.

## 먼저 수행할 실기 검증

- Unpack/scale: K256/512/2048, block offset mod4=0/2, code0~3, zero scale, row tails.
- MUL_MAT_ID: E256/S8, N1/2/4/8/13/184/665/2048, up/gate 2048→512, down 512→2048.
- Expert histogram: 균등, 일부expert 집중, 빈expert, nonzero/high expert IDs.
- 동일 raw activation/weights에서 baseline vs fused outputs; sampling을 통한 자유 생성 비교만 하지 않는다.
- Shader timestamps와 queue gaps, VRAM allocation, register/SLM occupancy 측정.
- 실제 모델의 fixed-prefix logits/KLD, 문맥 길이별 retrieval, 코드 빌드/테스트 회귀.
- multi-slot/stream 동시실행, request 취소, graph 재사용, buffer lifetime 검사.

## 아직 구현하지 않은 최적화

이 패키지는 XMX cooperative-matrix, subgroup별 세밀한 A750 tuning, GGUF load-time fused tensor 등록, native ggml graph hook, KV/KVarN, head 재양자화, multi-GPU layer split을 제공하지 않는다. Fused project가 현재 vendor-tuned matmul보다 빠르다고 가정하지 않는다. 특히 `NT4`는 보수적 시작값이며 prefill에서 더 큰 token tile/XMX를 쓰는 후속 설계가 필요할 수 있다.
