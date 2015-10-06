
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "comms.h"
#include "input.h"
#include "problem.h"
#include "allocate.h"
#include "halos.h"
#include "source.h"

#include "ocl_global.h"

/** \mainpage
* SNAP-MPI is a cut down version of the SNAP mini-app which allows us to
* investigate MPI decomposition schemes with various node-level implementations.
* In particular, this code will allow:
* \li Flat MPI
* \li Hybrid MPI+OpenMP (For CPU and larger core counts)
* \li OpenCL
*
* The MPI scheme used is KBA, expending into hybrid-KBA.
*/

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

    // Set dx, dy, dz, dt values
    globals.dx = globals.lx / (double)globals.nx;
    globals.dy = globals.ly / (double)globals.ny;
    globals.dz = globals.lz / (double)globals.nz;
    globals.dt = globals.tf / (double)globals.nsteps;

    // Broadcast the global variables
    broadcast_problem(&globals, rank);


    // Set up communication neighbours
    struct rankinfo local;
    setup_comms(&globals, &local);

    // Initlise the OpenCL
    struct context context;
    init_ocl(&context);


    // Allocate the problem arrays
    struct mem memory;
    allocate_memory(globals, local, &memory);
    struct halo halos;
    allocate_halos(&globals, &local, &halos);

    // Set up problem
    init_quadrature_weights(&globals, memory.quad_weights);
    calculate_cosine_coefficients(&globals, memory.mu, memory.eta, memory.xi);
    calculate_scattering_coefficients(&globals, memory.scat_coeff, memory.mu, memory.eta, memory.xi);
    init_material_data(&globals, memory.mat_cross_section);
    init_fixed_source(&globals, &local, memory.fixed_source);
    init_scattering_matrix(&globals, memory.mat_cross_section, memory.scattering_matrix);
    init_velocities(&globals, memory.velocities);
    init_velocity_delta(&globals, memory.velocities, memory.velocity_delta);

    // Time loop
    // TODO
    // swap angluar flux pointers


    // Outers
    calculate_dd_coefficients(&globals, memory.eta, memory.xi, memory.dd_i, memory.dd_j, memory.dd_k);
    calculate_denominator(&globals, &local, memory.dd_i, memory.dd_j, memory.dd_k, memory.mu, memory.mat_cross_section, memory.velocity_delta, memory.denominator);
    // Calculate outer source
    for (unsigned int i = 0; i < globals.ng*local.nx*local.ny*local.nz; i++)
        memory.scalar_flux_in[i] = 0.0;
    compute_outer_source(&globals, &local, memory.fixed_source, memory.scattering_matrix, memory.scalar_flux_in, memory.scalar_flux_moments, memory.outer_source);

    compute_inner_source(&globals, &local, memory.outer_source, memory.scattering_matrix, memory.scalar_flux_in, memory.scalar_flux_moments, memory.inner_source);


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
