
#pragma once

// Problem dimensions from input file
struct problem
{
    // Global grid size
    unsigned int nx, ny, nz;

    // Physical grid size
    double lx, ly, lz;

    // Width of spatial cells
    double dx, dy, dz;

    // Energy groups
    unsigned int ng;

    // Angles per octant
    // (3D assumed)
    unsigned int nang;

    // Number of expansion moments
    unsigned int nmom;

    // Number of computational moments = nmom*nmom
    unsigned int cmom;

    // Iteration limits (inner, outer)
    unsigned int iitm;
    unsigned int oitm;

    // Timestep details
    unsigned int nsteps;
    double tf;
    // Time domain stride
    double dt;

    // Convergence criteria
    double epsi;

    // Number of MPI tasks in each direction
    unsigned int npex, npey, npez;

    // KBA chunk size
    unsigned int chunk;
};


// Holds local information about tile size and MPI rank
struct rankinfo
{
    // My WORLD rank
    int rank;

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
