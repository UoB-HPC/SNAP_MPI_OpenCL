
#include "comms.h"

void check_mpi(const int err, const char *msg)
{
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI Error: %d. %s\n", err, msg);
        exit(err);
    }
}


void setup_comms(struct problem * problem, struct rankinfo * rankinfo)
{
    // Create the MPI Cartesian topology
    int dims[] = {problem->npex, problem->npey, problem->npez};
    int periods[] = {0, 0, 0};
    int mpi_err = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &snap_comms);
    check_mpi(mpi_err, "Creating MPI Cart");

    // Get my ranks in x, y and z
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rankinfo->rank);
    check_mpi(mpi_err, "Getting MPI rank");
    mpi_err = MPI_Cart_coords(snap_comms, rankinfo->rank, 3, rankinfo->ranks);
    check_mpi(mpi_err, "Getting Cart co-ordinates");

    // Note: The following assumes one tile per MPI rank
    // TODO: Change to allow for tiling

    // Calculate rankinfo sizes
    rankinfo->nx = problem->nx / problem->npex;
    rankinfo->ny = problem->ny / problem->npey;
    rankinfo->nz = problem->nz / problem->npez;

    // Calculate i,j,k lower and upper bounds in terms of problem grid
    rankinfo->ilb = rankinfo->ranks[0]     * rankinfo->nx;
    rankinfo->iub = (rankinfo->ranks[0]+1) * rankinfo->nx;
    rankinfo->jlb = rankinfo->ranks[1]     * rankinfo->ny;
    rankinfo->jub = (rankinfo->ranks[1]+1) * rankinfo->ny;
    rankinfo->klb = rankinfo->ranks[2]     * rankinfo->nz;
    rankinfo->kub = (rankinfo->ranks[2]+1) * rankinfo->nz;

    // Calculate neighbouring ranks
    calculate_neighbours(snap_comms, problem, rankinfo);
}

void finish_comms(void)
{
    int mpi_err = MPI_Finalize();
    check_mpi(mpi_err, "MPI_Finalize");
}

void calculate_neighbours(MPI_Comm comms,  struct problem * problem, struct rankinfo * rankinfo)
{
    int mpi_err;

    // Calculate my neighbours
    int coords[3];
    // x-dir + 1
    coords[0] = (rankinfo->ranks[0] == problem->npex - 1) ? rankinfo->ranks[0] : rankinfo->ranks[0] + 1;
    coords[1] = rankinfo->ranks[1];
    coords[2] = rankinfo->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &rankinfo->xup);
    check_mpi(mpi_err, "Getting x+1 rank");
    // x-dir - 1
    coords[0] = (rankinfo->ranks[0] == 0) ? rankinfo->ranks[0] : rankinfo->ranks[0] - 1;
    coords[1] = rankinfo->ranks[1];
    coords[2] = rankinfo->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &rankinfo->xdown);
    check_mpi(mpi_err, "Getting x-1 rank");
    // y-dir + 1
    coords[0] = rankinfo->ranks[0];
    coords[1] = (rankinfo->ranks[1] == problem->npey - 1) ? rankinfo->ranks[1] : rankinfo->ranks[1] + 1;
    coords[2] = rankinfo->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &rankinfo->yup);
    check_mpi(mpi_err, "Getting y+1 rank");
    // y-dir - 1
    coords[0] = rankinfo->ranks[0];
    coords[1] = (rankinfo->ranks[1] == 0) ? rankinfo->ranks[1] : rankinfo->ranks[1] - 1;
    coords[2] = rankinfo->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &rankinfo->ydown);
    check_mpi(mpi_err, "Getting y-1 rank");
    // z-dir + 1
    coords[0] = rankinfo->ranks[0];
    coords[1] = rankinfo->ranks[1];
    coords[2] = (rankinfo->ranks[2] == problem->npez - 1) ? rankinfo->ranks[2] : rankinfo->ranks[2] + 1;
    mpi_err = MPI_Cart_rank(comms, coords, &rankinfo->zup);
    check_mpi(mpi_err, "Getting z+1 rank");
    // z-dir - 1
    coords[0] = rankinfo->ranks[0];
    coords[1] = rankinfo->ranks[1];
    coords[2] = (rankinfo->ranks[2] == 0) ? rankinfo->ranks[2] : rankinfo->ranks[2] - 1;
    mpi_err = MPI_Cart_rank(comms, coords, &rankinfo->zdown);
    check_mpi(mpi_err, "Getting z-1 rank");
}
