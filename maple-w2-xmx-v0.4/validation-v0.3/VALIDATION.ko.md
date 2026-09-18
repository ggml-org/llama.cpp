# v0.3 검증 기록

## 새 코드에서 수행한 것

| 검사 | 결과 | 원문 |
|---|---|---|
| GCC Release CMake/CTest | 4/4 test target PASS | `ctest-cpu.log` |
| A16 CPU reference | 2,096,458 checks PASS | `cpu-check-counts.log` |
| A8 CPU reference | 1,888,002 checks PASS | `cpu-check-counts.log` |
| 새 MoE CPU reference | 97,428 checks PASS | `cpu-check-counts.log` |
| Python tests | 19 PASS | `python-tests.log` |
| GCC ASan+UBSan | 세 CPU reference 모두 PASS | `cpu-sanitizers.log` |
| Clang 17 strict CPU build | 새 MoE reference PASS | `clang-cpu-environment.log` |
| Quick/full runner 계획 | 3/7 MoE cases 출력 | `moe-quick-plan.json`, `moe-full-plan.json` |

CPU 테스트는 SwiGLU clipping, RNE, input/hidden 그룹, 정규화 route, expert 주소,
MoE 수학 조합을 검사한다. Python orchestration 테스트의 GPU 실행 mock은 **실제 실행이 아니라**
실패 처리·결과 합산의 unit test다.

## 아직 수행하지 않은 것

- 신규 `maple_moe.cpp`, `moe_compare.cpp`의 SYCL 컴파일.
- 신규 GPU epilogue/quant, full chain 정확성·성능·in-order/out-of-order 실행.
- PowerShell 및 cmd wrapper의 Windows 실행.
- 실제 Vulkan/Level Zero interop, llama 서버 end-to-end TG.
- A8 모델 품질과 speculative acceptance.

환경에 `icpx`, `icx`, `sycl-ls`, `pwsh`, `powershell`이 없다는 조회 결과를 보존했다.
CPU pass를 GPU pass나 동기화 비용 감소의 실측으로 간주하지 않는다.

## 사용자 v0.2 증거는 별도

`../evidence/user-v0.2-run.log`는 사용자의 A750 측정이다. 6 cases / 85 summaries PASS,
DPAS s2/s8 probe 36 cases / 288 outputs PASS다. 새 v0.3의 GPU 성공 증거가 아니다.
Source requests/수치 probe와 최종 device ISA 관찰도 구분한다. 해당 log에 ISA dump는 없다.
