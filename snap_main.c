
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "comms.h"
#include "input.h"
#include "problem.h"
#include "allocate.h"
#include "halos.h"


int main(int argc, char **argv)
{
    int mpi_err = MPI_Init(&argc, &argv);
    check_mpi(mpi_err, "MPI_Init");

    int rank, size;
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    check_mpi(mpi_err, "Getting MPI rank");

    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    check_mpi(mpi_err, "Getting MPI size");

    struct problem globals;

    if (rank == 0)
    {
        // Check for two files on CLI
        if (argc != 3)
        {
            fprintf(stderr, "Usage: ./snap snap.in snap.out\n");
            exit(-1);
        }
        read_input(argv[1], &globals);
        if ((globals.npex * globals.npey * globals.npez) != size)
        {
            fprintf(stderr, "Input error: wanted %d ranks but executing with %d\n", globals.npex*globals.npey*globals.npez, size);
            exit(-1);
        }
        check_decomposition(&globals);

    }
    // Broadcast the global variables
    broadcast_problem(&globals, rank);


    // Set up communication neighbours
    struct rankinfo local;
    setup_comms(&globals, &local);


    // Allocate the problem arrays
    struct mem memory;
    allocate_memory(globals, local, &memory);
    struct halo halos;
    allocate_halos(&globals, &local, &halos);

    // Set up problem
    calculate_cosine_coefficients(&globals, memory.mu, memory.eta, memory.xi);
    calculate_scattering_coefficients(&globals, memory.scat_coeff, memory.mu, memory.eta, memory.xi);

    // Halo exchange routines

    // Loop over octants
    int istep, jstep, kstep;
    for (unsigned int OmZ = 0; OmZ < 2; OmZ++)
        for (unsigned int OmY = 0; OmY < 2; OmY++)
            for (unsigned int OmX = 0; OmX < 2; OmX++)
            {
                istep = (OmX == 0)? -1 : 1;
                jstep = (OmY == 0)? -1 : 1;
                kstep = (OmZ == 0)? -1 : 1;



            }
    // Receive data from neighbours

    // Sweep chunk
    // Send data to neighbours



    free_halos(&globals, &halos);
    free_memory(&memory);

    finish_comms();
}
