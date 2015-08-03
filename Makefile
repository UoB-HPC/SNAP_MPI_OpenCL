
ifndef COMPILER
	MESSAGE=select a compiler to compiler in OpenMP, e.g. make COMPILER=INTEL
endif


OMP_INTEL	= -openmp
OMP_CRAY	= 
OMP_GNU		= -fopenmp

OMP=$(OMP_$(COMPILER))

CFLAGS_			= -O3
CFLAGS_INTEL	= -O3 -std=c99
CFLAGS_CRAY		= -O3
CFLAGS_GNU		= -O3 -std=c99

MPI_COMPILER 	= mpicc

snap: \
	snap_main.c \
	input.c \
	allocate.c \
	halos.c \
	comms.c
	$(MPI_COMPILER) $^ $(CFLAGS_$(COMPILER)) $(OMP) -o $@
