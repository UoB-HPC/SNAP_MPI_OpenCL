
#include "convergence.h"

static double tolr=1.0E-12;

bool inner_convergence(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct memory * memory
    )
{
    double diffs[problem->ng];
    for (unsigned int g = 0; g < problem->ng; g++)
        diffs[g] = -DBL_MAX;

    // Calculate the maximum difference across each sub-domain for each group
    for (unsigned int k = 0; k < rankinfo->nz; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
                for (unsigned int g = 0; g < problem->ng; g++)
                {
                    double new = memory->scalar_flux[SCALAR_FLUX_INDEX(g,i,j,k,problem->ng,rankinfo->nx,rankinfo->ny)];
                    double old = memory->old_inner_scalar_flux[SCALAR_FLUX_INDEX(g,i,j,k,problem->ng,rankinfo->nx,rankinfo->ny)];
                    if (fabs(old) > tolr)
                        diffs[g] = fmax(fabs(new / old - 1.0), diffs[g]);
                    else
                        diffs[g] = fmax(fabs(new - old), diffs[g]);
                }

    // Do an AllReduce for each group to work out global maximum difference
    bool result = false;
    for (unsigned int g = 0; g < problem->ng; g++)
    {
        double recv;
        MPI_Allreduce(diffs+g, &recv, 1, MPI_DOUBLE, MPI_MAX, snap_comms);
        diffs[g] = recv;
        result &= diffs[g] <= problem->epsi;
    }
    return result;
}
