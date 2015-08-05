
#pragma once

#include "global.h"

// Memory access patterns
#define SCAT_COEFF_INDEX(a,m,o,nang,cmom) ((a)+((nang)*(m))+((nang)*(cmom)*o))
#define FIXED_SOURCE_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))
#define SCATTERING_MATRIX_INDEX(m,g1,g2,nmom,ng) ((m)+((nmom)*(g1))+((nmom)*(ng)*(g2)))

void init_quadrature_weights(const struct problem * global, double * restrict quad_weights);
void calculate_cosine_coefficients(const struct problem * global, double * restrict mu, double * restrict eta, double * restrict xi);
void calculate_scattering_coefficients(const struct problem * global, double * restrict scat_coef, const double * restrict mu, const double * restrict eta, const double * restrict xi);
void init_material_data(const struct problem * global, double * restrict mat_cross_section);
void init_fixed_source(const struct problem * global, const struct rankinfo * local, double * restrict fixed_source);
void init_scattering_matrix(const struct problem * global, const double * restrict mat_cross_section, double * restrict scattering_matrix);

void calculate_dd_coefficients(const struct problem * global, const double * restrict eta, const double * restrict xi, double * restrict dd_i, double * restrict dd_j, double * restrict dd_k);
