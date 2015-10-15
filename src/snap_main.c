
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "comms.h"
#include "input.h"
#include "problem.h"
#include "allocate.h"
#include "halos.h"
#include "source.h"
#include "sweep.h"

#include "ocl_global.h"
#include "ocl_buffers.h"
#include "ocl_sweep.h"

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

extern double wtime(void);

int main(int argc, char **argv)
{
    int mpi_err = MPI_Init(&argc, &argv);
    check_mpi(mpi_err, "MPI_Init");

    double setup_time = wtime();

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
    struct memory memory;
    allocate_memory(&problem, &rankinfo, &memory);
    struct halo halos;
    allocate_halos(&problem, &rankinfo, &halos);

    // Set up problem
    init_quadrature_weights(&problem, &context, &buffers);
    calculate_cosine_coefficients(&problem, &context, &buffers, memory.mu, memory.eta, memory.xi);
    calculate_scattering_coefficients(&problem, &context, &buffers, memory.mu, memory.eta, memory.xi);
    init_material_data(&problem, &context, &buffers, memory.mat_cross_section);
    init_fixed_source(&problem, &rankinfo, &context, &buffers);
    init_scattering_matrix(&problem, &context, &buffers, memory.mat_cross_section);
    init_velocities(&problem, &context, &buffers);

    struct plane* planes;
    unsigned int num_planes;
    init_planes(&planes, &num_planes, &rankinfo);
    copy_planes(planes, num_planes, &context, &buffers);

    // Save the scalar_flux_moments buffer size for repeated zero'ing every timestep
    size_t scalar_moments_buffer_size;
    if (problem.cmom-1 == 0)
        scalar_moments_buffer_size = problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;
    else
        scalar_moments_buffer_size = (problem.cmom-1)*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz;

    // Zero out the angular flux buffers
    for (int oct = 0; oct < 8; oct++)
    {
        zero_buffer(&context, buffers.angular_flux_in[oct], problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        zero_buffer(&context, buffers.angular_flux_out[oct], problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
    }

    cl_int err = clFinish(context.queue);
    check_ocl(err, "Finish queue at end of setup");

    setup_time = wtime() - setup_time;
    printf("Setup took %lfs\n", setup_time);

    double simulation_time = wtime();

    //----------------------------------------------
    // Timestep loop
    //----------------------------------------------
    for (unsigned int t = 0; t < problem.nsteps; t++)
    {
        // Zero out the scalar flux and flux moments
        zero_buffer(&context, buffers.scalar_flux, problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        zero_buffer(&context, buffers.scalar_flux_moments, scalar_moments_buffer_size);

        //----------------------------------------------
        // Outers
        //----------------------------------------------
        for (unsigned int o = 0; o < problem.oitm; o++)
        {
            init_velocity_delta(&problem, &context, &buffers);
            calculate_dd_coefficients(&problem, &context, &buffers);
            calculate_denominator(&problem, &rankinfo, &context, &buffers);

            compute_outer_source(&problem, &rankinfo, &context, &buffers);

            //----------------------------------------------
            // Inners
            //----------------------------------------------
            for (unsigned int i = 0; i < problem.iitm; i++)
            {
                compute_inner_source(&problem, &rankinfo, &context, &buffers);

                // Zero out the incoming boundary fluxes
                zero_buffer(&context, buffers.flux_i, problem.nang*problem.ng*rankinfo.ny*rankinfo.nz);
                zero_buffer(&context, buffers.flux_j, problem.nang*problem.ng*rankinfo.nx*rankinfo.nz);
                zero_buffer(&context, buffers.flux_k, problem.nang*problem.ng*rankinfo.nx*rankinfo.ny);

                // Sweep each octant
                // Octant 1
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(0, -1, -1, -1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 2
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(1, +1, -1, -1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 3
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(2, -1, +1, -1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 4
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(3, +1, +1, -1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 5
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(4, -1, -1, +1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 6
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(5, +1, -1, +1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 7
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(6, -1, +1, +1, p, planes, &problem, &rankinfo, &context, &buffers);
                // Octant 8
                for (unsigned int p = 0; p < num_planes; p++)
                    sweep_plane(7, +1, +1, +1, p, planes, &problem, &rankinfo, &context, &buffers);
            }
            //----------------------------------------------
            // End of Inners
            //----------------------------------------------
        }
        //----------------------------------------------
        // End of Outers
        //----------------------------------------------

        // swap angluar flux pointers
    }
    //----------------------------------------------
    // End of Timestep
    //----------------------------------------------

    err = clFinish(context.queue);
    printf("%d\n", err);

    simulation_time = wtime() - simulation_time;
    printf("Simulation took %lfs\n", simulation_time);

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
