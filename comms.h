
#pragma once

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "problem.h"

// Holds local information about tile size and MPI rank
struct rankinfo
{

    // My MPI Cartesian co-ordinate ranks
    int ranks[3];

    // Local grid size
    unsigned int nx, ny, nz;

    // Global grid corners of MPI partition
    unsigned int ilb, iub;
    unsigned int jlb, jub;
    unsigned int klb, kub;

    // My neighbours
    int xup, xdown;
    int yup, ydown;
    int zup, zdown;
};

// Cartesian communicator
MPI_Comm snap_comms;

void check_mpi(const int err, const char *msg);
void setup_comms(struct problem * global, struct rankinfo * local);
void finish_comms(void);

void calculate_neighbours(MPI_Comm comms,  struct problem * global, struct rankinfo * local);
