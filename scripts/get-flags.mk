ifeq '' '$(findstring clang,$(shell $(GF_CC) --version))'
	GF_CC_IS_GCC = 1
	GF_CC_VER := $(shell { $(GF_CC) -dumpfullversion 2>/dev/null; echo; $(GF_CC) -dumpversion; } | awk -F. '/./ { printf("%02d%02d%02d", $$1, $$2, $$3); exit }')
else
	GF_CC_IS_CLANG = 1
	ifeq '' '$(findstring Apple,$(shell $(GF_CC) --version))'
		GF_CC_IS_LLVM_CLANG = 1
	else
		GF_CC_IS_APPLE_CLANG = 1
	endif
	GF_CC_VER := \
		$(shell $(GF_CC) --version | sed -n 's/^.* version \([0-9.]*\).*$$/\1/p' \
		| awk -F. '{ printf("%02d%02d%02d", $$1, $$2, $$3) }')
endif

ifeq ($(GF_CC_IS_CLANG), 1)
	# clang options
	GF_CFLAGS   = -Wunreachable-code-break -Wunreachable-code-return
	GF_CXXFLAGS = -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi

	ifneq '' '$(and $(GF_CC_IS_LLVM_CLANG),$(filter 1,$(shell expr $(GF_CC_VER) \>= 030800)))'
		GF_CFLAGS += -Wdouble-promotion
	endif
	ifneq '' '$(and $(GF_CC_IS_APPLE_CLANG),$(filter 1,$(shell expr $(GF_CC_VER) \>= 070300)))'
		GF_CFLAGS += -Wdouble-promotion
	endif
else
	# gcc options
	GF_CFLAGS   = -Wdouble-promotion
	GF_CXXFLAGS = -Wno-array-bounds

	ifeq ($(shell expr $(GF_CC_VER) \>= 070100), 1)
		GF_CXXFLAGS += -Wno-format-truncation
	endif
	ifeq ($(shell expr $(GF_CC_VER) \>= 080100), 1)
		GF_CXXFLAGS += -Wextra-semi
	endif
endif
