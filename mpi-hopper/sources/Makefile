# on Franklin and Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS. The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, check the NERSC documentation.
# This makefile is intended for the GNU C compiler. On Franklin and Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU: type "module swap PrgEnv-pgi PrgEnv-gnu"
# Your code must compile (with GCC) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = cc 
OPT = -O2
CFLAGS = -Wall $(OPT) -std=gnu99 -funroll-loops -funit-at-a-time -mfpmath=sse -fstrict-aliasing -ftree-vectorize -ffast-math 
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt

targets = dnsmat 
objects = main.o matrix_mul_hopper.o 

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

dnsmat: matrix_mul_hopper.o main.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
