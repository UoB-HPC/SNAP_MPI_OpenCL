
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
