# 검증 결과 — v0.4 Vulkan architecture port v1

**이 파일의 PASS는 A750 실측이 아니다.** oneAPI compiler/SDK와 `/dev/dri`가 없는 Linux CPU 환경에서 실행한 결과다.

| 검사 | 실제 결과 |
|---|---|
| 업로드 v0.4 내부 SHA256 목록 | 전 항목 일치 |
| GCC 14.2 Release CTest | **8/8 PASS** |
| Clang 17 Release CTest | **8/8 PASS** |
| AddressSanitizer + UndefinedBehaviorSanitizer + leak check | **8/8 PASS** |
| Python unittest | **48 tests PASS** |
| 일반 C++로 신규 비교기 host syntax 검사 | GCC/Clang PASS, test-double 헤더 사용 |
| 원래 T1 A8·SwiGLU/hidden quant 수학 구간 | 소스 원문 동일 |
| Intel SYCL / ESIMD 컴파일·링크 | **미실행** |
| A750 신규 T2/T4·fork/join·성능 | **미실행** |
| llama-server E2E·모델 품질 | **미실행** |

## 테스트 수의 의미

- 기존 TQ2 CPU checks: 2,096,458.
- 기존 W2A8 CPU checks: 1,888,002.
- 기존 MoE reference checks: 97,428.
- 기존 grouping reference checks: 15,476,059.
- 신규 policy/tile/quant/timestamp CPU checks: 1,397,115.
- 실제 host enqueue 코드의 mocked-DAG checks: 2,868.
- 실제 신규 token-reuse kernel 본문의 scalar-DPAS 에뮬레이션 checks: 268,806.

숫자는 assertion 수다. GPU 실험 횟수나 GPU kernel PASS 수가 아니다. scalar-DPAS 에뮬레이터는 실제 `src/maple_w2a8_grouped.cpp`를 실행하므로 주소·slot·scale·split·tail·canary 검증에 쓰였지만, Intel compiler의 DPAS operand lowering이나 register spill, 실행시간을 검증하지 못한다.

Mock-DAG는 실제 `src/maple_moe.cpp`와 `src/expert_grouping.cpp`의 host submission을 실행한다. device lambda는 실행하지 않고, root deps→group/quant→gate→hidden→down→sum이 연결되는지, default host wait가 없는지, 잘못된 workspace가 첫 submit 전에 거부되는지 검사한다.

## 별도의 실제 GPU 검증 절차

1. `build_architecture_and_test.cmd -Suite smoke`.
2. native s2/s8 ISA probe, static grouping/descriptor probe, RNE/tiny/large/nonfinite quantizer probe를 통과시킨다.
3. T2/T4 candidate와 T1 baseline의 모든 출력/quant plane을 비교한다.
4. 같은 GPU q/scale을 쓰는 stage-local CPU oracle와 quantizer 자체 audit를 모두 통과시킨다.
5. Q sweep에서 paired wall·GPU span·group 비용·overlap을 확인한다.
6. 마지막에 llama.cpp E2E와 모델 품질을 검증한다.

이 패키지에는 성능 예측치를 실측값처럼 넣지 않았다. full-plan.json의 78 cases는 dry-run 구성이고 측정 결과가 아니다. 구버전 `validation*`/`evidence` 디렉터리는 과거 버전의 기록으로 그대로 보존했다.

## 로그

`gcc-*`, `clang-*`, `sanitize-*`, `python-tests.log`, `cpu-check-counts.log`, `host-driver-syntax.log`, `environment.log`, `preserved-source-regions.log`, `patch-apply-check.log`.
