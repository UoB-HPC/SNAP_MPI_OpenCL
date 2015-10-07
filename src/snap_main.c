
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
#include "ocl_buffers.h"

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

    struct problem problem;

    if (rank == 0)
    {
        // Check for two files on CLI
        if (argc != 3)
        {
            fprintf(stderr, "Usage: ./snap snap.in snap.out\n");
            exit(-1);
        }
        read_input(argv[1], &problem);
        if ((problem.npex * problem.npey * problem.npez) != size)
        {
            fprintf(stderr, "Input error: wanted %d ranks but executing with %d\n", problem.npex*problem.npey*problem.npez, size);
            exit(-1);
        }
        check_decomposition(&problem);

    }

    // Set dx, dy, dz, dt values
    problem.dx = problem.lx / (double)problem.nx;
    problem.dy = problem.ly / (double)problem.ny;
    problem.dz = problem.lz / (double)problem.nz;
    problem.dt = problem.tf / (double)problem.nsteps;

    // Broadcast the global variables
    broadcast_problem(&problem, rank);


    // Set up communication neighbours
    struct rankinfo rankinfo;
    setup_comms(&problem, &rankinfo);

    // Initlise the OpenCL
    struct context context;
    init_ocl(&context);
    struct buffers buffers;
    check_device_memory_requirements(&problem, &rankinfo, &context);
    allocate_buffers(&problem, &rankinfo, &context, &buffers);


    // Allocate the problem arrays
    struct mem memory;
    allocate_memory(&problem, &rankinfo, &memory);
    struct halo halos;
    allocate_halos(&problem, &rankinfo, &halos);

    // Set up problem
    init_quadrature_weights(&problem, &context, &buffers);
    calculate_cosine_coefficients(&problem, &context, &buffers);
    // calculate_scattering_coefficients(&problem, memory.scat_coeff, memory.mu, memory.eta, memory.xi);
    init_material_data(&problem, memory.mat_cross_section);
    init_fixed_source(&problem, &rankinfo, memory.fixed_source);
    init_scattering_matrix(&problem, memory.mat_cross_section, memory.scattering_matrix);
    init_velocities(&problem, memory.velocities);

    // Time loop
    // TODO
    // swap angluar flux pointers


    // Outers
    init_velocity_delta(&problem, memory.velocities, memory.velocity_delta);
    //calculate_dd_coefficients(&problem, memory.eta, memory.xi, memory.dd_i, memory.dd_j, memory.dd_k);
    //calculate_denominator(&problem, &rankinfo, memory.dd_i, memory.dd_j, memory.dd_k, memory.mu, memory.mat_cross_section, memory.velocity_delta, memory.denominator);
    // Calculate outer source
    for (unsigned int i = 0; i < problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz; i++)
        memory.scalar_flux_in[i] = 0.0;
    compute_outer_source(&problem, &rankinfo, memory.fixed_source, memory.scattering_matrix, memory.scalar_flux_in, memory.scalar_flux_moments, memory.outer_source);

    compute_inner_source(&problem, &rankinfo, memory.outer_source, memory.scattering_matrix, memory.scalar_flux_in, memory.scalar_flux_moments, memory.inner_source);


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



    free_halos(&problem, &halos);
    free_memory(&memory);

    release_context(&context);
    finish_comms();
}
