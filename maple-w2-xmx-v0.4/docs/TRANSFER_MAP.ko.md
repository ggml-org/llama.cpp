# Vulkan 연구 결과 → XMX v0.4 변경 대응

1. 작은 direct 작업에서 전역 reuse가 손해였음 → gate/down 별도 T1/T2/T4, 기본 T1. Vulkan 시간이나 WG crossover를 XMX의 자동값으로 옮기지 않음.
2. static bucket이 persistent보다 나았음 → 이미 존재하는 v0.4 static grouping 유지. persistent scheduler는 추가하지 않음.
3. 여러 assignment가 같은 expert를 사용해야 weight 재사용이 생김 → expert boundary를 넘지 않는 token descriptor + T2/T4 B-load reuse. old mapped job order만 바꾸던 구현과 구분.
4. group/quant가 독립 작업임 → explicit fork/join DAG, input quant producer 분리, producer 공유 API.
5. Q8 한 code 차이가 downstream을 크게 바꿀 수 있음 → 실제 GPU q/scale 기반 projection oracle + 별도 quantizer audit, raw dump와 freeze-input.
6. CPU/GPU rounding contract를 구분해야 함 → XMX의 RNE 유지. Vulkan half-away의 111을 XMX expected value로 복사하지 않음.
7. stage 합과 GPU elapsed가 다름 → min/max span + interval union + overlap/idle. 부모 의존성만 검사하고 독립 stage의 timestamp overlap은 허용.

## 변하지 않은 기본 연산

`tests/test_grouping_suite.py::test_glu_and_a8_math_unchanged`가 v0.3에서 보존된 v0.4 연산 구간과 원문 일치를 검사한다. 이 검사는 문자열/소스 수준의 불변성이지 새 binary ISA 동일성 보증은 아니다.

- signed s2 packing, original 66-byte block scale semantics
- 원래 `dpas<8,1>` direct/grouped T1 경로
- Pair gate/up 및 K-block별 scaling 순서
- RNE input/hidden quantization
- gate upper-only clamp와 up ±clamp
- split reduction 순서와 final original top-k slot sum

신규 T2/T4 파일은 이 연산을 token별로 반복하되 B/scale load를 공유한다. CPU 테스트는 actual file body를 scalar ESIMD/DPAS model로 실행한다. Intel compiler lowering과 A750 register pressure/latency는 별도 검증해야 한다.

## 외부 API 근거

Khronos SYCL reference의 event, queue, handler 설명을 확인했다. 같은 backend의 profiling timebase 및 explicit depends_on을 사용한다. in-order queue는 앞선 command에 암묵적 의존성을 부여하므로 fork/join 옵션만 켜서는 실제 overlap이 생기지 않는다.

- https://github.khronos.org/SYCL_Reference/iface/event.html
- https://github.khronos.org/SYCL_Reference/iface/queue.html
- https://github.khronos.org/SYCL_Reference/iface/command-group-handler.html
- https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd/examples/README.md

인터넷 문서는 API 의미를 확인한 근거다. 이 패키지의 실제 컴파일·속도 결과를 대신하지 않는다.
