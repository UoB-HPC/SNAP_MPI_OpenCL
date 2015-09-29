
#include "source.h"

void compute_outer_source(
    const struct problem * global,
    const struct rankinfo * local,
    const double * restrict fixed_source,
    const double * restrict scattering_matrix,
    const double * restrict scalar_flux,
    const double * restrict scalar_flux_moments,
    double * restrict outer_source
    )
{
    for (unsigned int k = 0; k < local->nz; k++)
        for (unsigned int j = 0; j < local->ny; j++)
            for (unsigned int i = 0; i < local->nx; i++)
                for (unsigned int g = 0; g < global->ng; g++)
                {
                    // Set first moment to the fixed source
                    outer_source[OUTER_SOURCE_INDEX(0,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]
                        = fixed_source[FIXED_SOURCE_INDEX(g,i,j,k,global->ng,local->nx,local->ny)];

                    // Loop over groups and moments to compute out-of-group scattering
                    for (unsigned int g2 = 0; g2 < global->ng; g2++)
                    {
                        if (g == g2)
                            continue;
                        // Compute scattering source
                        outer_source[OUTER_SOURCE_INDEX(0,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]
                            += scattering_matrix[SCATTERING_MATRIX_INDEX(0,g2,g,global->nmom,global->ng)]
                            * scalar_flux[SCALAR_FLUX_INDEX(g2,i,j,k,global->ng,local->nx,local->ny)];
                        // Other moments
                        unsigned int mom = 1;
                        for (unsigned int l = 0; l < global->nmom; l++)
                        {
                            for (unsigned int m = 0; m < 2*l+1; m++)
                            {
                                outer_source[OUTER_SOURCE_INDEX(mom,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]
                                    += scattering_matrix[SCATTERING_MATRIX_INDEX(l,g2,g,global->nmom,global->ng)]
                                    * scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX(mom-1,g2,i,j,k,global->cmom,global->ng,local->nx,local->ny)];
                                mom += 1;
                            }
                        }
                    }
                }
}


void compute_inner_source(
    const struct problem * global,
    const struct rankinfo * local,
    const double * restrict outer_source,
    const double * restrict scattering_matrix,
    const double * restrict scalar_flux,
    const double * restrict scalar_flux_moments,
    double * restrict inner_source
    )
{
    for (unsigned int k = 0; k < local->nz; k++)
        for (unsigned int j = 0; j < local->ny; j++)
            for (unsigned int i = 0; i < local->nx; i++)
                for (unsigned int g = 0; g < global->ng; g++)
                {
                    // Set first moment to outer source plus scattering contribution of scalar flux
                    inner_source[INNER_SOURCE_INDEX(0,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]
                        = outer_source[OUTER_SOURCE_INDEX(0,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]
                        + (scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g,global->nmom,global->ng)]
                        * scalar_flux[SCALAR_FLUX_INDEX(g,i,j,k,global->ng,local->nx,local->ny)]);

                    // Set other moments similarly based on scalar flux moments
                    unsigned int mom = 1;
                    for (unsigned int l = 0; l < global->nmom; l++)
                    {
                        for (unsigned int m = 0; m < 2*l+1; m++)
                        {
                            inner_source[INNER_SOURCE_INDEX(mom,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]
                                = outer_source[OUTER_SOURCE_INDEX(mom,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)] +
                                + (scattering_matrix[SCATTERING_MATRIX_INDEX(l,g,g,global->nmom,global->ng)]
                                * scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX(mom-1,g,i,j,k,global->cmom,global->ng,local->nx,local->ny)]);
                            mom += 1;
                        }
                    }
                }
}
