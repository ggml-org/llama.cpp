HEXAGON_SDK_PATH=/opt/qcom/Hexagon_SDK/6.2.0.1

TARGET=libggmlop_skel.so

INCS=-I${HEXAGON_SDK_PATH}/incs -I${HEXAGON_SDK_PATH}/libs/qprintf/inc -I${HEXAGON_SDK_PATH}/incs/stddef -I${HEXAGON_SDK_PATH}/ipc/fastrpc/incs -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc -I${HEXAGON_SDK_PATH}/utils/examples -I${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/inc -I${HEXAGON_SDK_PATH}/libs/atomic/inc -I${HEXAGON_SDK_PATH}/utils/sim_utils/inc

CFLAGS=-mv75 -c -Ofast -Wall -Wstrict-prototypes -fno-zero-initialized-in-bss -fdata-sections -fpic -D__V_DYNAMIC__ -mhvx -mhvx-length=128B ${INCS}

LDFLAGS=-mv75 -Wl,--defsym=ISDB_TRUSTED_FLAG=2 -Wl,--defsym=ISDB_SECURE_FLAG=2 -Wl,--no-threads -fpic -shared -Wl,-Bsymbolic -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -lc -Wl,-soname=${TARGET}

SRCS = ggmlop_cdsp.c  ggmlop_cdsp_skel.c
OBJS = $(patsubst %.c, %.o, $(SRCS))

ALL:$(OBJS)
		${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang ${LDFLAGS} -o ${TARGET} -Wl,--start-group ${OBJS} -Wl,--end-group
		@ls -l ${TARGET}

%.o:%.c
		@echo "${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang ${CFLAGS} -D__FILENAME__=\"$<\" -o $@ -c $< "
		${HEXAGON_SDK_PATH}/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang ${CFLAGS} -D__FILENAME__=\"$<\" -o $@ -c $<
		@echo "\n"

clean:
	rm -f *.o
