include(CheckCSourceRuns)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(riscv)" AND GGML_CPU_RISCV64_SPACEMIT)
    set(SMT_MARCH_STR "-march=rv64gcv_zfh_zvfh_zba_zicbop")
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15)
        string(APPEND SMT_MARCH_STR "_xsmtvdotii")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${SMT_MARCH_STR}")

    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1, i4\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1, i8\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S8)
    check_c_source_compiles("int main() {__asm__ volatile(\"vfwmadot v2, v0, v1, fp16\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VFWMADOT_FP16)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot.hp v2, v0, v1, v0, 0, i4\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VFMADOT_S4)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot.hp v2, v0, v1, v0, 0, i8\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VFMADOT_S8)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot1 v2, v0, v1\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOTN)
    check_c_source_compiles("int main() {__asm__ volatile(\"vpack.vv v2, v0, v1, 2\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK)
    check_c_source_compiles("int main() {__asm__ volatile(\"vnspack.vv v2, v0, v1, 2\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK)
    unset(CMAKE_REQUIRED_FLAGS)

    list(APPEND RISCV64_SPACEMIT_IME_SPEC "")
    if (SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
        set(RISCV64_SPACEMIT_IME_SPEC "RISCV64_SPACEMIT_IME1")
    endif()

    if (SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4 AND SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK AND SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK)
        list(APPEND RISCV64_SPACEMIT_IME_SPEC "RISCV64_SPACEMIT_IME2")
    endif()

    message("RISCV64_SPACEMIT_IME_SPEC: ${RISCV64_SPACEMIT_IME_SPEC}")
endif()
