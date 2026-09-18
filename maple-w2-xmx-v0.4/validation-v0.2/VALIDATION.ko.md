# v0.2 검증 결과

**2026-09-17 / Linux CPU-only 환경. 이 파일은 GPU 검증 결과가 아니다.**

| 항목 | 결과 |
|---|---|
| 기존 W2A16 CPU reference | 2,096,458 checks 통과 |
| 신규 W2A8 CPU reference | 1,888,002 checks 통과 |
| CTest | 3/3 targets 통과 |
| Python 추출기·실행기 unit tests | 11 tests 통과 |
| GCC ASan + UBSan + leak check | A16/A8 reference 모두 통과 |
| Clang 17 CPU 빌드·실행 | A16/A8 reference 모두 통과 |
| Python AST 검사·suite dry run | quick 6 / full 17 case 구성 확인 |

CPU reference 검사는 signed INT2 패킹, 독립 DPAS 배치 에뮬레이션,
기존 TQ2 scale 보존, A8 그룹/RNE, expert 주소, broadcast/per-selection,
split-K의 수학을 검사한다. **2,096,458/1,888,002는 assertion/check 횟수이지 GPU 실험 횟수가 아니다.**

## 실행하지 못한 검사

icx/icpx/dpcpp, PowerShell, GPU 장치가 없다. 따라서 다음은 미검증이다.

- SYCL source compilation/linking 및 Windows 빌드 스크립트 실행.
- A750 native s2/s8 DPAS probe, GPU 수치 결과, 생성 ISA, GPU 속도.
- Vulkan↔SYCL 전환과 서버 연결, Maple 전체 품질/TG.

`build_and_test.ps1`은 로컬에서 이 GPU 검증을 시작하기 위한 스크립트다.
probe/수치 검사 실패 시 suite가 실패로 종료되고 로그를 보존한다.
성공해도 이는 kernel implementation 검사이며 A8 모델 품질 허용을 뜻하지 않는다.

## 재현 로그

- `cmake-configure.log`, `cmake-build.log`, `ctest-cpu.log`
- `cpu-sanitizers.log`
- `clang-cpu-and-environment.log`
- `python-syntax.log`
- `quick-plan.json`, `full-plan.json`
- `manifest.json`

CPU source를 빌드한 결과로 SYCL source의 컴파일 성공을 주장하지 않는다.
