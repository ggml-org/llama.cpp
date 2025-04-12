#following vars already defined in CMakeLists.txt
#HTP_ARCH_VERSION=v79
#DEBUG_FLAG=-DNDEBUG -Wall
#HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/6.2.0.1

HEXAGON_COMPUTE=compute${HTP_ARCH_VERSION}
HEXAGON_CC=${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang
HEXAGON_CXX=${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang

TARGET=libggmlop-skel.so

$(info HEXAGON_SDK_PATH:${HEXAGON_SDK_PATH})
$(info HTP_ARCH_VERSION:${HTP_ARCH_VERSION})
$(info DEBUG_FLAG:${DEBUG_FLAG})
$(info HEXAGON_COMPUTE:${HEXAGON_COMPUTE})

INCS=-I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/libs/qprintf/inc -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/inc -I${HEXAGON_SDK_PATH}/libs/atomic/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/posix -I${HEXAGON_SDK_PATH}/rtos/qurt/${HEXAGON_COMPUTE}/include/qurt/

CFLAGS=-m${HTP_ARCH_VERSION} -c -Ofast -Wall -Wstrict-prototypes -fno-zero-initialized-in-bss -fdata-sections -fpic -D__V_DYNAMIC__ -mhvx -mhvx-length=128B ${INCS} -fno-finite-math-only

LDFLAGS=-m${HTP_ARCH_VERSION} -Wl,--defsym=ISDB_TRUSTED_FLAG=2 -Wl,--defsym=ISDB_SECURE_FLAG=2 -Wl,--no-threads -fpic -shared -Wl,-Bsymbolic -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -lc -Wl,-soname=${TARGET}

#SRCS = $(wildcard *.c)
SRCS = ggml-dsp.c skel.c add.c  mulmat.c
OBJS = $(patsubst %.c, %.o, $(SRCS))

ALL:$(OBJS)
		${HEXAGON_CC} ${LDFLAGS} -o ${TARGET} -Wl,--start-group ${OBJS} -Wl,--end-group
		@ls -l ${TARGET}
		/bin/cp -fv ${TARGET} ../../../../out/android/bin/
		/bin/cp -fv ${TARGET} ../../../../out/android/bin/libggmlop-skel${HTP_ARCH_VERSION}.so
		/bin/rm -f *.so

%.o:%.c
		@echo "${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<"
		${HEXAGON_CC} ${CFLAGS} ${DEBUG_FLAG} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

clean:
	rm -f *.o
