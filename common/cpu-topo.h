#pragma once

#include <vector>

// Internal helpers for math CPU counting on x86_64 Linux, exposed for tests.
// See common.cpp and tests/test-cpu-topo.cpp.
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)

// x2APIC right-shift for the core id from the SMT level. -1 = no topology
// leaves, 0 = no SMT (each logical processor is its own core), else shift bits.
int cpu_smt_shift(int has_topology, int has_smt_level, unsigned smt_eax, unsigned smt_ebx);

// Distinct physical cores: skip efficiency cores, core id = x2apic >> shift.
int cpu_count_distinct_cores(const std::vector<unsigned> & x2apic,
                             const std::vector<char> & is_efficiency,
                             int shift);

#endif
