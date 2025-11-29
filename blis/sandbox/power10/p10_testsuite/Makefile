BLIS_PATH := ../../..

BLIS_INC  := $(BLIS_PATH)/include/power10
BLIS_LIB  := $(BLIS_PATH)/lib/power10/libblis.a

CC := gcc
LINKER := $(CC)

CFLAGS := -I $(BLIS_INC)
LDFLAGS := -lpthread -lm

OBJS      := $(patsubst %.c,%.o, $(wildcard *.c))
PERF_OBJS := performance.o
COR_OBJS  := correctness.o cast_funcs.o

all: performance correctness

$(OBJS): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

performance: $(PERF_OBJS)
	$(LINKER) $(PERF_OBJS) $(BLIS_LIB) -o ./gather_perf.x $(LDFLAGS) 

correctness: $(COR_OBJS)
	$(LINKER) $(COR_OBJS) $(BLIS_LIB) -o ./test_correctness.x $(LDFLAGS) 

csv_clean:
	rm -rf *.csv

clean: 
	rm -rf *.x *.o