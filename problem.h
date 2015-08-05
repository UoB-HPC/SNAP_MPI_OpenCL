
#pragma once

// Problem dimensions from input file
struct problem
{
    // Global grid size
    unsigned int nx, ny, nz;

    // Physical grid size
    double lx, ly, lz;

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

    // Convergence criteria
    double epsi;

    // Number of MPI tasks in each direction
    unsigned int npex, npey, npez;

    // KBA chunk size
    unsigned int chunk;
};

// Memory access patterns
#define SCAT_COEFF_INDEX(a,m,o,nang,cmom) (a)+((nang)*(m))+((nang)*(cmom)*o)

void calculate_cosine_coefficients(const struct problem * global, double * restrict mu, double * restrict eta, double * restrict xi);
void calculate_scattering_coefficients(const struct problem * global, double * restrict scat_coef, const double * restrict mu, const double * restrict eta, const double * restrict xi);
void init_material_data(const struct problem * global, double * restrict mat_cross_section);

