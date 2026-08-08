DEBUG_FLAG=-DNDEBUG -Wall

HEXAGON_COMPUTE=compute${HTP_ARCH_VERSION}
HEXAGON_CC=${HEXAGON_TOOLS_PATH}/Tools/bin/hexagon-clang
HEXAGON_CXX=${HEXAGON_TOOLS_PATH}/Tools/bin/hexagon-clang

TARGET=libggmldsp-skel.so

$(info HEXAGON_SDK_PATH:${HEXAGON_SDK_PATH})
$(info HEXAGON_TOOLS_PATH:${HEXAGON_TOOLS_PATH})
$(info HTP_ARCH_VERSION:${HTP_ARCH_VERSION})
$(info DEBUG_FLAG:${DEBUG_FLAG})
$(info HEXAGON_COMPUTE:${HEXAGON_COMPUTE})

INCS=-I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/libs/qprintf/inc -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/inc -I${HEXAGON_SDK_PATH}/libs/atomic/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/posix -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/qurt/ -I. -I../../ -I../../../include -I${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/target/hexagon/include

CFLAGS=-m${HTP_ARCH_VERSION} -c -O3 -ffast-math -Wall -Wstrict-prototypes -fno-zero-initialized-in-bss -fdata-sections -fpic -D__V_DYNAMIC__ -mhvx -mhvx-length=128B -mhmx ${INCS} -fno-finite-math-only -fno-vectorize

LDFLAGS=-m${HTP_ARCH_VERSION} -Wl,--defsym=ISDB_TRUSTED_FLAG=2 -Wl,--defsym=ISDB_SECURE_FLAG=2 -Wl,--no-threads -fpic -shared -Wl,-Bsymbolic -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -lc -Wl,-soname=${TARGET}


# Qualcomm execute_op kernels (binary-ops.c, matmul-ops.c, etc.) live in this directory.
# entry.c dispatches all ops through the execute_op path (algotype=29).
SRCS = ggml_dsp_skel.c entry.c work-queue.c dma-queue.c hmx-queue.c htp-tensor.c binary-ops.c unary-ops.c sum-rows-ops.c softmax-ops.c act-ops.c rope-ops.c set-rows-ops.c get-rows-ops.c cpy-ops.c repeat-ops.c argsort-ops.c ssm-conv.c cumsum-ops.c fill-ops.c concat-ops.c diag-ops.c solve-tri-ops.c gated-delta-net-ops.c pad-ops.c matmul-ops.c flash-attn-ops.c im2col-ops.c
OBJS = $(patsubst %.c, %.o, $(patsubst %.cpp, %.o, $(SRCS)))

ALL:$(OBJS)
		${HEXAGON_CC} ${LDFLAGS} -o ${TARGET} -Wl,--start-group ${OBJS} -Wl,--end-group
		@ls -l ${TARGET}
		@rm -f *.o
		@rm -f ggml_dsp.h  ggml_dsp_skel.c  ggml_dsp_stub.c


%.o:%.c
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

# flash-attn-ops.c triggers a Hexagon LLVM 19.0.07 backend bug:
#   PromoteFloatResult #0: t23: f16 = freeze t10
#   fatal error: Do not know how to promote this operator's result!
# Workaround: lower optimization to -O2 (avoids the freeze-generating pass).
flash-attn-ops.o: flash-attn-ops.c
		@echo "${HEXAGON_CC} $(filter-out -O3,$(CFLAGS)) -O2 ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} $(filter-out -O3,$(CFLAGS)) -O2 ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

%.o:%.S
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

%.o:%.cpp
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

clean:
	rm -f *.o
	/bin/rm -f *.so
