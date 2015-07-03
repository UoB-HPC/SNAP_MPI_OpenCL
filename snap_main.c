
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "input.h"
#include "problem.h"
#include "allocate.h"
#include "halos.h"

void check_mpi(const int err, const char *msg)
{
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI Error: %d. %s\n", err, msg);
        exit(err);
    }
}


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

    // Create the MPI Cartesian topology
    MPI_Comm snap_comms;
    int dims[] = {globals.npex, globals.npey, globals.npez};
    int periods[] = {0, 0, 0};
    mpi_err = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &snap_comms);
    check_mpi(mpi_err, "Creating MPI Cart");

    // Get my ranks in x, y and z
    struct rankinfo local;
    mpi_err = MPI_Cart_coords(snap_comms, rank, 3, local.ranks);
    check_mpi(mpi_err, "Getting Cart co-ordinates");

    // Note: The following assumes one tile per MPI rank
    // TODO: Change to allow for tiling

    // Calculate local sizes
    local.nx = globals.nx / globals.npex;
    local.ny = globals.ny / globals.npey;
    local.nz = globals.nz / globals.npez;

    // Calculate i,j,k lower and upper bounds in terms of global grid
    local.ilb = local.ranks[0]*local.nx;
    local.iub = (local.ranks[0]+1)*local.nx;
    local.jlb = local.ranks[1]*local.ny;
    local.jub = (local.ranks[1]+1)*local.ny;
    local.klb = local.ranks[2]*local.nz;
    local.kub = (local.ranks[2]+1)*local.nz;


    // Allocate the problem arrays
    struct mem memory;
    allocate_memory(globals, local, &memory);
    struct halo halos;
    allocate_halos(&globals, &local, &halos);

    // Set up problem

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

                // Calculate neighbours
                int idown, jdown, kdown;
                idown = local.ranks[0] + istep;
                jdown = local.ranks[1] + jstep;
                kdown = local.ranks[2] + kstep;

                // If off the processor grid, use your own rank
                if (idown < 0) idown = local.ranks[0];
                if (idown >= globals.npex) idown = local.ranks[0];

                if (jdown < 0) jdown = local.ranks[1];
                if (jdown >= globals.npey) jdown = local.ranks[1];

                if (kdown < 0) kdown = local.ranks[2];
                if (kdown >= globals.npez) kdown = local.ranks[2];
    
                printf("i am %d %d %d, %d\n", local.ranks[0], local.ranks[1], local.ranks[2], rank);

                // Send to X neighbour
                if (idown != local.ranks[0])
                {
                    int xrank;
                    int coords[3] = {idown, local.ranks[1], local.ranks[2]};
                    MPI_Cart_rank(snap_comms, coords, &xrank);
                    printf("my x neighbour is %d %d %d, %d\n", idown, local.ranks[1], local.ranks[2], xrank);
                }


            }
    // Receive data from neighbours

    // Sweep chunk
    // Send data to neighbours



    free_halos(&halos);
    free_memory(&memory);

    mpi_err = MPI_Finalize();
    check_mpi(mpi_err, "MPI_Finalize");
}
