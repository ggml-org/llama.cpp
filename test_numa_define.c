#ifdef GGML_NUMA_MIRROR
#ifdef __cplusplus
extern "C" {
#endif
int check_numa_mirror_defined() { return 1; }
#ifdef __cplusplus
}
#endif
#else
#ifdef __cplusplus
extern "C" {
#endif
int check_numa_mirror_defined() { return 0; }
#ifdef __cplusplus
}
#endif
#endif
