
CC	= gcc
TARGET_ARCH = -m64 -mvsx

TGTS	= exp

KERNEL	= bli_gemm_opt_8x4.o

CFLAGS  = -DUTEST -std=gnu99 -ggdb3 -Wall
CFLAGS  += -O3

all: $(TGTS)

exp: exp.o $(KERNEL)

clean:
	rm -f $(TGTS) *.o

