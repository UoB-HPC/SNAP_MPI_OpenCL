
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
};
