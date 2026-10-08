cmake_minimum_required(VERSION 3.16)
include("${CMAKE_CURRENT_LIST_DIR}/../ggml/src/ggml-sycl/xmx-gather.cmake")

if (DEFINED INVALID_MODE)
    ggml_sycl_xmx_gather_targets("${INVALID_MODE}" "bmg-g21" actual)
    return()
endif()

function(check_targets mode arch expected)
    ggml_sycl_xmx_gather_targets("${mode}" "${arch}" actual)
    if (NOT actual STREQUAL expected)
        message(FATAL_ERROR "${mode} '${arch}': expected '${expected}', got '${actual}'")
    endif()
endfunction()

# Unknown targets must not inherit support from a similar name or a range endpoint.
foreach(arch acm-g10 acm-g11 acm-g12 dg2 xe-hpg xe_hpg 12.55.8 12.56.5 12.57
             mtl-h arl-h tgl adl-p ats-m150 ats-m75 0x56A0 dg1:acm-g10 xe
             xe3 xe-hpc 0 off bmg bmg-g210 bmg-g21:acm-g10 pvc-unknown lnl-m-extra)
    check_targets(AUTO "${arch}" OFF)
endforeach()
foreach(arch bmg-g21 xe2-hpg pvc lnl-m)
    check_targets(AUTO "${arch}" "${arch}")
    check_targets(OFF "${arch}" OFF)
endforeach()

check_targets(AUTO "" JIT)
check_targets(AUTO " ; , " JIT)
check_targets(AUTO "bmg-g21;BMG_G21" bmg-g21)
check_targets(ON "" JIT)
check_targets(OFF "" OFF)
check_targets(auto "BMG_G21" bmg-g21)
check_targets(AUTO "acm-g10,bmg-g21" bmg-g21)
check_targets(AUTO "bmg-g21,acm-g10" bmg-g21)
check_targets(AUTO " ACM_G10;BMG_G21,xe2_hpg PVC;lnl-m " "bmg-g21,xe2-hpg,pvc,lnl-m")
# Explicit ON preserves unsupported targets so it cannot silently select stubs.
check_targets(on "ACM_G10,unknown_future" "acm-g10,unknown-future")
check_targets(ON "dg1:acm-g10;BMG_G21" "dg1:acm-g10,bmg-g21")
check_targets(off "acm-g10,bmg-g21" OFF)

foreach(mode invalid TRUE 1)
    execute_process(COMMAND "${CMAKE_COMMAND}" "-DINVALID_MODE=${mode}"
                    -P "${CMAKE_CURRENT_LIST_FILE}"
                    RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if (rc EQUAL 0 OR NOT "${out}${err}" MATCHES "AUTO.*ON.*OFF")
        message(FATAL_ERROR "invalid mode '${mode}' was not rejected with a policy error: ${out}${err}")
    endif()
endforeach()
message(STATUS "SYCL XMX gather target policy passed")
