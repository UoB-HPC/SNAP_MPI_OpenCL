
#pragma once

#include "global.h"

// Memory access patterns
#define SCAT_COEFF_INDEX(a,m,o,nang,cmom) (a)+((nang)*(m))+((nang)*(cmom)*o)

void calculate_cosine_coefficients(const struct problem * global, double * restrict mu, double * restrict eta, double * restrict xi);
void calculate_scattering_coefficients(const struct problem * global, double * restrict scat_coef, const double * restrict mu, const double * restrict eta, const double * restrict xi);
void init_material_data(const struct problem * global, double * restrict mat_cross_section);

