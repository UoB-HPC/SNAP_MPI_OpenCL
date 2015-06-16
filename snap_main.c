
#include <stdio.h>
#include <mpi.h>

#include "problem.h"

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

    printf("%d %d\n", rank, size);

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
    }

    mpi_err = MPI_Finalize();
    check_mpi(mpi_err, "MPI_Finalize");
}
