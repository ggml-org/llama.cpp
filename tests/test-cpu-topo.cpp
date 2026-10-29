// Regression for math CPU counting on hybrid x86_64 CPUs without SMT (#28625).
//
// Calls the real production helpers in common/cpu-topo.h:
// - cpu_smt_shift: the SMT shift decision, fed recorded CPUID register values.
// - cpu_count_distinct_cores: the core id dedup, fed recorded/constructed topologies.
//
// Part 1 calls the public API common_cpu_get_num_math() to make sure the patch
// does not break the non-hybrid fallback path. Parts 2 and 3 use recorded or
// constructed values, NOT live CPUID reads - live verification on a no-SMT hybrid
// part needs real hardware, see #28625.

#include "common.h"

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include "../common/cpu-topo.h"
#endif

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // Part 1: public API smoke (real common.cpp code, all platforms).
    {
        int32_t math = common_cpu_get_num_math();
        printf("common_cpu_get_num_math() = %d\n", math);
        assert(math > 0);
    }

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    // Part 2: cpu_smt_shift decision (real helper, recorded CPUID regs).
    // Per Intel SDM leaf 0xB/0x1F: EAX[4:0] = SMT mask width (the shift), EBX[15:0]
    // = threads per core, ECX[15:8] = level type (1 = SMT). ECX[7:0] is the level
    // number, NOT the shift - reading it (an earlier draft) overcounts on SMT-on.
    struct smt_case {
        const char * name;
        int has_topology;
        int has_smt_level;
        unsigned smt_eax; // EAX at SMT level; [4:0] = shift
        unsigned smt_ebx; // EBX at SMT level; [15:0] = threads per core
        int expected;
    };
    std::vector<smt_case> sc = {
        {"no topology leaves",      0, 0, 0x00, 0, -1},
        {"topology, no SMT level",  1, 0, 0x00, 0,  0},
        {"SMT, TPC=1 (no SMT)",     1, 1, 0x00, 1,  0}, // width 0
        {"SMT, TPC=2 (2-way)",      1, 1, 0x01, 2,  1}, // width 1 - the fix
        {"SMT, TPC=4 (4-way)",      1, 1, 0x02, 4,  2}, // width 2
        {"SMT, EBX=0 (malformed)",  1, 1, 0x01, 0,  0}, // defensive
    };

    // Part 3: cpu_count_distinct_cores dedup (real helper, recorded topologies).
    struct topo_case {
        const char * name;
        std::vector<unsigned> x2apic;
        std::vector<char> is_efficiency; // 1 = efficiency/LP-E, skipped
        int shift;
        int expected;
    };
    std::vector<topo_case> tc = {
        // 255H: 6P (logical 0,1,10-13) + 8E (2-9) + 2LP-E (14,15), no SMT, shift 0.
        // Recorded on the native host; old loop gave 3, want 6.
        {"255H 6P+8E+2LPE no-SMT",
         {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
         {0,0,1,1,1,1,1,1,1,1, 0, 0, 0, 0, 1, 1}, 0, 6},
        // 12700H: 6P x2 SMT + 8E, 20 logical, SMT on, shift 1. Old gave 6, want 6.
        {"12700H 6P+8E SMT-on",
         {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19},
         {0,0,0,0,0,0,0,0,0,0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, 1, 6},
        // 265K (#28625): 8P no SMT + E, 20 logical, shift 0. Old gave 4, want 8.
        {"265K 8P+E no-SMT",
         {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19},
         {0,0,0,0,0,0,0,0,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 0, 8},
        // Generic 8c/16t, all P, SMT on, shift 1. Old gave 8, want 8 (no regression).
        {"8c/16t SMT-on",
         {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
         {0,0,0,0,0,0,0,0,0,0, 0, 0, 0, 0, 0, 0}, 1, 8},
        // Adversarial: 6 contiguous P-cores + 10 E, no SMT, shift 0. Old gave 3, want 6.
        {"6P contiguous + 10E no-SMT",
         {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
         {0,0,0,0,0,0,1,1,1,1, 1, 1, 1, 1, 1, 1}, 0, 6},
        {"all efficiency", {0,1,2,3}, {1,1,1,1}, 0, 0},
        {"single P-core", {7}, {0}, 0, 1},
        {"empty", {}, {}, 0, 0},
    };

    int failures = 0;
    for (const auto & c : sc) {
        int got = cpu_smt_shift(c.has_topology, c.has_smt_level, c.smt_eax, c.smt_ebx);
        bool ok = (got == c.expected);
        printf("[%s] smt %-24s: got=%d expected=%d\n", ok ? "PASS" : "FAIL", c.name, got, c.expected);
        if (!ok) failures++;
    }
    for (const auto & c : tc) {
        int got = cpu_count_distinct_cores(c.x2apic, c.is_efficiency, c.shift);
        bool ok = (got == c.expected);
        printf("[%s] topo %-26s: got=%d expected=%d\n", ok ? "PASS" : "FAIL", c.name, got, c.expected);
        if (!ok) failures++;
    }
    if (failures > 0) {
        printf("%d case(s) FAILED\n", failures);
        return 1;
    }
    printf("all cases passed\n");
#endif
    return 0;
}
