
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
