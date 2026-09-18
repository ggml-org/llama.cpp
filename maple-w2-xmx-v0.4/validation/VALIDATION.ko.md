# v0.4 검증 상태

작성 환경: Linux x86-64, GCC 14.2, Clang 17. Intel oneAPI/SYCL compiler와 A750은 없다.

## 실제로 수행한 검사

| 검사 | 결과 |
|---|---|
| 기존 A16 CPU reference | 2,096,458 checks PASS |
| 기존 A8 CPU reference | 1,888,002 checks PASS |
| 기존 MoE CPU reference | 97,428 checks PASS |
| 새 grouping CPU reference | 15,476,059 checks PASS |
| CTest | 5/5 PASS |
| Python unittest | 34개 PASS |
| GCC ASan+UBSan | 위 네 CPU reference 모두 PASS |
| Clang grouping CPU reference | PASS |
| A8 DPAS/unpack/scale 구간과 GLU 원본 동일성 | 소스 excerpt 비교 PASS |

새 grouping 검사에는 독립 stable_sort와 cursor counting sort, GPU 구조를 모사한 chunk-scan을
대조했다. expert/ID 범위, invalid bucket, 128 lane tail, 중복 ID, 모두 동일 expert, 모든 expert,
원래 출력 주소와 split scratch 주소의 단일 기록, quant/route 불변성을 확인한다.
이 check 수는 CPU assertion 수이며 GPU 테스트 횟수가 아니다.

새 comparator와 grouping translation unit은 작성용 임시 SYCL 모형 선언으로 C++ 문법도
검사했다. **이는 실제 SYCL 헤더·컴파일러에 대한 검사나 device code 생성이 아니다.**
그 임시 모형은 제품 소스에 포함하지 않는다.

## 로컬에서 실행해야 하는 검사

**SYCL 컴파일, GPU mapping probe, full GPU A/B, GPU 수치 검사, 시간 측정과 Windows PowerShell 실행은 미실행.**
따라서 v0.4 GPU PASS나 가속 배율은 아직 없다. ZIP에 성능 결과를 합성해 넣지 않았다.

실행 스크립트는 signed INT2/INT8 probe 뒤, 48개 grouping probe를 수행한다.
실제 chain에서는 두 ID snapshot의 모든 GPU output과 quant plane을 비교하고,
CPU reference는 최대 3개 token의 모든 선택 expert/output에 적용한다.
Grouping은 마지막 timed call의 매핑까지 재확인한다. 검사가 실패하면 성능 집계를 중단한다.

이전 버전의 검증 기록은 `validation-v0.1/`, `validation-v0.2/`, `validation-v0.3/`에 별도로 보존했다.
그 기록을 신규 v0.4 GPU 성공으로 해석하면 안 된다.
