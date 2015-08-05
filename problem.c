
#include "problem.h"
#include <math.h>

void calculate_cosine_coefficients(
    const struct problem * global,
    double * restrict mu,
    double * restrict eta,
    double * restrict xi
    )
{
    double dm = 1.0 / global->nang;

    mu[0] = 0.5 * dm;
    eta[0] = 1.0 - 0.5 * dm;
    double t = mu[0] * mu[0] + eta[0] * eta[0];
    xi[0] = sqrt(1.0 - t);

    for (unsigned int a = 1; a < global->nang; a++)
    {
        mu[a] = mu[a-1] + dm;
        eta[a] = eta[a-1] - dm;
        t = mu[a] * mu[a] + eta[a] * eta[a];
        xi[a] = sqrt(1.0 - t);
    }
}

void calculate_scattering_coefficients(
    const struct problem * global,
    double * restrict scat_coef,
    const double * restrict mu,
    const double * restrict eta,
    const double * restrict xi
    )
{
    // (mu*eta*xi)^l starting at 0
    for (int kd = 0; kd < 2; kd++)
    {
        double ks = (kd == 1) ? 1.0 : -1.0;
        for (int jd = 0; jd < 2; jd++)
        {
            double js = (jd == 1) ? 1.0 : -1.0;
            for (int id = 0; id < 2; id++)
            {
                double is = (id == 1) ? 1.0 : -1.0;
                int oct = 4*kd + 2*jd + id;
                // Init first moment
                for (unsigned int a = 0; a < global->nang; a++)
                    scat_coef[SCAT_COEFF_INDEX(a,0,oct,global->nang,global->cmom)] = 1.0;
                // Init other moments
                int mom = 1;
                for (int l = 1; l < global->nmom; l++)
                {
                    for (int m = 0; m < 2*l+1; m++)
                    {
                        for (unsigned int a = 0; a < global->nang; a++)
                        {
                            scat_coef[SCAT_COEFF_INDEX(a,mom,oct,global->nang,global->cmom)] = pow(is*mu[a], 2.0*l-1.0) * pow(ks*xi[a]*js*eta[a], m);
                        }
                        mom += 1;
                    }
                }
            }
        }
    }
}

void init_material_data(
    const struct problem * global,
    double * restrict mat_cross_section
    )
{
    mat_cross_section[0] = 1.0;
    for (unsigned int g = 1; g < global->ng; g++)
    {
        mat_cross_section[g] = mat_cross_section[g-1] + 0.01;
    }
}

void init_fixed_source(
    const struct problem * global,
    const struct rankinfo * local,
    double * restrict fixed_source
    )
{
    // Source everywhere, set at strength 1.0
    // This is src_opt == 0 in original SNAP
    for(unsigned int k = 0; k < local->nz; k++)
        for(unsigned int j = 0; j < local->ny; j++)
            for(unsigned int i = 0; i < local->nx; i++)
                for(unsigned int g = 0; g < global->ng; g++)
                    fixed_source[FIXED_SOURCE_INDEX(g,i,j,k,global->ng,local->nx,local->ny)] = 1.0;
}

void init_scattering_matrix(
        const struct problem * global,
        const double * restrict mat_cross_section,
        double * restrict scattering_matrix
    )
{
    // 10% up scattering
    // 20% in group scattering
    // 70% down scattering
    // First and last group, no up/down scattering
    for (unsigned int g = 0; g < global->ng; g++)
    {
        if (global->ng == 1)
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,global->nmom,global->ng)] = mat_cross_section[g] * 0.5;
            break;
        }

        scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g,global->nmom,global->ng)] = 0.2 * 0.5 * mat_cross_section[g];

        if (g > 0)
        {
            double t = 1.0 / (double)(g);
            for (unsigned int g2 = 0; g2 < g; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,global->nmom,global->ng)] = 0.1 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,global->nmom,global->ng)] = 0.3 * 0.5 * mat_cross_section[0];
        }

        if (g < (global->ng) - 1)
        {
            double t = 1.0 / (double)(global->ng - g - 1);
            for (unsigned int g2 = g + 1; g2 < global->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,global->nmom,global->ng)] = 0.7 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,global->ng-1,global->ng-1,global->nmom,global->ng)] = 0.9 * 0.5 * mat_cross_section[global->ng-1];
        }
    }

    // Set scattering moments (up to 4)
    // Second moment 10% of first, subsequent half of previous
    if (global->nmom > 1)
    {
        for (unsigned int g1 = 0; g1 < global->ng; g1++)
        {
            for (unsigned int g2 = 0; g2 < global->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(1,g1,g2,global->nmom,global->ng)] = 0.1 * scattering_matrix[SCATTERING_MATRIX_INDEX(0,g1,g2,global->nmom,global->ng)];
                for (unsigned int m = 2; m < global->nmom; m++)
                {
                    scattering_matrix[SCATTERING_MATRIX_INDEX(m,g1,g2,global->nmom,global->ng)] = 0.5 * scattering_matrix[SCATTERING_MATRIX_INDEX(m-1,g1,g2,global->nmom,global->ng)];
                }
            }
        }
    }
}
