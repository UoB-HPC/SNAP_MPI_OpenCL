
#pragma once

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "global.h"
#include "allocate.h"
#include "ocl_global.h"
#include "ocl_buffers.h"


// Cartesian communicator
MPI_Comm snap_comms;

void check_mpi(const int err, const char *msg);
void setup_comms(struct problem * problem, struct rankinfo * rankinfo);
void finish_comms(void);

void calculate_neighbours(MPI_Comm comms,  struct problem * problem, struct rankinfo * rankinfo);


void recv_boundaries(const int octant, const int istep, const int jstep, const int kstep, struct problem * problem, struct rankinfo * rankinfo, struct memory * memory, struct context * context, struct buffers * buffers);
void send_boundaries(const int octant, const int istep, const int jstep, const int kstep, struct problem * problem, struct rankinfo * rankinfo, struct memory * memory, struct context * context, struct buffers * buffers);
