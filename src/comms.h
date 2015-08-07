
#pragma once

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "global.h"


// Cartesian communicator
MPI_Comm snap_comms;

void check_mpi(const int err, const char *msg);
void setup_comms(struct problem * global, struct rankinfo * local);
void finish_comms(void);

void calculate_neighbours(MPI_Comm comms,  struct problem * global, struct rankinfo * local);
