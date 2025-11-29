#if defined(__i386) || defined(_X86)
ARCH_X86
#endif

#if defined(__x86_64__) || defined(__amd64__)
ARCH_X86_64
#endif

#if defined(__arm__)
ARCH_ARM
#endif

#if defined(__aarch64__)
ARCH_AARCH64
#endif
