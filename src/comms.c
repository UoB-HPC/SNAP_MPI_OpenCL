
#include "comms.h"

void check_mpi(const int err, const char *msg)
{
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI Error: %d. %s\n", err, msg);
        exit(err);
    }
}


void setup_comms(struct problem * problem, struct rankinfo * local)
{
    // Create the MPI Cartesian topology
    int dims[] = {problem->npex, problem->npey, problem->npez};
    int periods[] = {0, 0, 0};
    int mpi_err = MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &snap_comms);
    check_mpi(mpi_err, "Creating MPI Cart");

    // Get my ranks in x, y and z
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &local->rank);
    check_mpi(mpi_err, "Getting MPI rank");
    mpi_err = MPI_Cart_coords(snap_comms, local->rank, 3, local->ranks);
    check_mpi(mpi_err, "Getting Cart co-ordinates");

    // Note: The following assumes one tile per MPI rank
    // TODO: Change to allow for tiling

    // Calculate local sizes
    local->nx = problem->nx / problem->npex;
    local->ny = problem->ny / problem->npey;
    local->nz = problem->nz / problem->npez;

    // Calculate i,j,k lower and upper bounds in terms of problem grid
    local->ilb = local->ranks[0]     * local->nx;
    local->iub = (local->ranks[0]+1) * local->nx;
    local->jlb = local->ranks[1]     * local->ny;
    local->jub = (local->ranks[1]+1) * local->ny;
    local->klb = local->ranks[2]     * local->nz;
    local->kub = (local->ranks[2]+1) * local->nz;

    // Calculate neighbouring ranks
    calculate_neighbours(snap_comms, problem, local);
}

void finish_comms(void)
{
    int mpi_err = MPI_Finalize();
    check_mpi(mpi_err, "MPI_Finalize");
}

void calculate_neighbours(MPI_Comm comms,  struct problem * problem, struct rankinfo * local)
{
    int mpi_err;

    // Calculate my neighbours
    int coords[3];
    // x-dir + 1
    coords[0] = (local->ranks[0] == problem->npex - 1) ? local->ranks[0] : local->ranks[0] + 1;
    coords[1] = local->ranks[1];
    coords[2] = local->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &local->xup);
    check_mpi(mpi_err, "Getting x+1 rank");
    // x-dir - 1
    coords[0] = (local->ranks[0] == 0) ? local->ranks[0] : local->ranks[0] - 1;
    coords[1] = local->ranks[1];
    coords[2] = local->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &local->xdown);
    check_mpi(mpi_err, "Getting x-1 rank");
    // y-dir + 1
    coords[0] = local->ranks[0];
    coords[1] = (local->ranks[1] == problem->npey - 1) ? local->ranks[1] : local->ranks[1] + 1;
    coords[2] = local->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &local->yup);
    check_mpi(mpi_err, "Getting y+1 rank");
    // y-dir - 1
    coords[0] = local->ranks[0];
    coords[1] = (local->ranks[1] == 0) ? local->ranks[1] : local->ranks[1] - 1;
    coords[2] = local->ranks[2];
    mpi_err = MPI_Cart_rank(comms, coords, &local->ydown);
    check_mpi(mpi_err, "Getting y-1 rank");
    // z-dir + 1
    coords[0] = local->ranks[0];
    coords[1] = local->ranks[1];
    coords[2] = (local->ranks[2] == problem->npez - 1) ? local->ranks[2] : local->ranks[2] + 1;
    mpi_err = MPI_Cart_rank(comms, coords, &local->zup);
    check_mpi(mpi_err, "Getting z+1 rank");
    // z-dir - 1
    coords[0] = local->ranks[0];
    coords[1] = local->ranks[1];
    coords[2] = (local->ranks[2] == 0) ? local->ranks[2] : local->ranks[2] - 1;
    mpi_err = MPI_Cart_rank(comms, coords, &local->zdown);
    check_mpi(mpi_err, "Getting z-1 rank");
}
