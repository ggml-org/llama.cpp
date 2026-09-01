#include "common.h"

#include <cstdio>
#include <cctype>
#include <cstdlib>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/sysctl.h>
#endif

static int n_fail = 0;

static void check(const char * what, int32_t got, int32_t want) {
    if (got != want) {
        fprintf(stderr, "FAIL: %s: got %d, want %d\n", what, got, want);
        n_fail++;
    }
}

#if defined(__APPLE__) && defined(__MACH__)
static void check_live_apple() {
    int32_t nlevels = 0;
    size_t  len     = sizeof(nlevels);
    if (sysctlbyname("hw.nperflevels", &nlevels, &len, NULL, 0) != 0 || nlevels <= 0) {
        return;
    }

    for (int32_t i = 0; i < nlevels; i++) {
        char key[64];
        char name[64] = {};
        size_t name_len = sizeof(name) - 1;
        snprintf(key, sizeof(key), "hw.perflevel%d.name", i);
        if (sysctlbyname(key, name, &name_len, NULL, 0) != 0) {
            return;
        }
        std::string lname = name;
        for (auto & c : lname) {
            c = (char) tolower((unsigned char) c);
        }
        if (lname.find("efficiency") != std::string::npos) {
            return;
        }
    }

    int32_t n_physical = 0;
    len = sizeof(n_physical);
    if (sysctlbyname("hw.physicalcpu", &n_physical, &len, NULL, 0) != 0 || n_physical <= 0) {
        return;
    }
    check("live: no Efficiency cluster, default == hw.physicalcpu", common_cpu_get_num_physical_cores(), n_physical);
}
#endif

int main() {
    using levels_t = std::vector<std::pair<std::string, int32_t>>;

    check("M4 Max (12P + 4E)", common_cpu_count_apple_perf_cores(levels_t{{"Performance", 12}, {"Efficiency", 4}}), 12);

    check("M5 Max (6S + 12P)", common_cpu_count_apple_perf_cores(levels_t{{"Super", 6}, {"Performance", 12}}), 18);

    check("Super + Performance + Efficiency", common_cpu_count_apple_perf_cores(levels_t{{"Super", 2}, {"Performance", 4}, {"Efficiency", 6}}), 6);

    check("lower case names", common_cpu_count_apple_perf_cores(levels_t{{"super", 6}, {"PERFORMANCE", 12}}), 18);

    check("unknown name only", common_cpu_count_apple_perf_cores(levels_t{{"Ultra", 8}}), 0);
    check("unknown name is skipped", common_cpu_count_apple_perf_cores(levels_t{{"Super", 6}, {"Ultra", 8}}), 6);

    check("empty", common_cpu_count_apple_perf_cores(levels_t{}), 0);
    check("zero cores", common_cpu_count_apple_perf_cores(levels_t{{"Performance", 0}}), 0);
    check("negative cores", common_cpu_count_apple_perf_cores(levels_t{{"Performance", -1}}), 0);

#if defined(__APPLE__) && defined(__MACH__)
    check_live_apple();
#endif

    if (n_fail > 0) {
        return EXIT_FAILURE;
    }
    printf("OK\n");
    return EXIT_SUCCESS;
}
