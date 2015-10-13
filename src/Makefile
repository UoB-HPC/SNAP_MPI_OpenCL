
ifndef COMPILER
	MESSAGE=select a compiler to compiler in OpenMP, e.g. make COMPILER=INTEL
endif

ifeq ($(shell uname), Darwin)
	OCL = -framework OpenCL
else
	OCL = -lOpenCL
endif

CFLAGS_			= -O3
CFLAGS_INTEL	= -O3 -std=c99
CFLAGS_CRAY		= -O3
CFLAGS_GNU		= -O3 -std=c99 -lm

MPI_COMPILER 	= mpicc


snap: \
	snap_main.c \
	input.c \
	allocate.c \
	halos.c \
	comms.c \
	problem.c \
	source.c \
	timer.c \
	sweep.c \
	ocl_init.c \
	ocl_buffers.c \
	ocl_sweep.c
	$(MPI_COMPILER) $^ $(CFLAGS_$(COMPILER)) $(OCL) $(OPTIONS) -o $@

kernels/%.h:	kernels/%.cl
	./stringify_opencl $< $@


KERNEL_HEADER_FILES=\
	kernels/calc_velocity_delta.h \
	kernels/calc_dd_coeff.h \
	kernels/calc_denominator.h \
	kernels/zero_buffer.h \
	kernels/outer_source.h \
	kernels/inner_source.h \
	kernels/sweep_plane.h

ocl_init.c:	$(KERNEL_HEADER_FILES)
	@touch ocl_init.c

