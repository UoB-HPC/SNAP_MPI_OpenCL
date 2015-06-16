
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

    // Iteration limits (inner, outer, time)
    unsigned int iitm;
    unsigned int oitm;
    unsigned int nsteps;

    // Convergence criteria
    double epsi;
};
