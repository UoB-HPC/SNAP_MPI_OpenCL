
/** @file */

#pragma once

#include "global.h"

/** @defgroup MEM Memory access patterns
* \brief Macros for indexing multi-dimensional arrays
* @{*/

/** \brief Index for scattering coefficient array */
#define SCAT_COEFF_INDEX(a,m,o,nang,cmom) ((a)+((nang)*(m))+((nang)*(cmom)*o))

/** \brief Index for fixed source array */
#define FIXED_SOURCE_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))

/** \brief Index for scattering matrix array */
#define SCATTERING_MATRIX_INDEX(m,g1,g2,nmom,ng) ((m)+((nmom)*(g1))+((nmom)*(ng)*(g2)))

/** \brief Index for transport denominator array */
#define DENOMINATOR_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
/**@}*/


void init_quadrature_weights(const struct problem * global, double * restrict quad_weights);
void calculate_cosine_coefficients(const struct problem * global, double * restrict mu, double * restrict eta, double * restrict xi);
void calculate_scattering_coefficients(const struct problem * global, double * restrict scat_coef, const double * restrict mu, const double * restrict eta, const double * restrict xi);
void init_material_data(const struct problem * global, double * restrict mat_cross_section);
void init_fixed_source(const struct problem * global, const struct rankinfo * local, double * restrict fixed_source);
void init_scattering_matrix(const struct problem * global, const double * restrict mat_cross_section, double * restrict scattering_matrix);
void init_velocities(const struct problem * global, double * restrict velocities);
void init_velocity_delta(const struct problem * global, const double * restrict velocities, double * restrict velocity_delta);

void calculate_dd_coefficients(const struct problem * global, const double * restrict eta, const double * restrict xi, double * restrict dd_i, double * restrict dd_j, double * restrict dd_k);
void calculate_denominator(const struct problem * global, const struct rankinfo * local, const double * restrict dd_i, const double * restrict dd_j, const double * restrict dd_k, const double * restrict mu, const double * restrict mat_cross_section, const double * restrict velocity_delta, double * restrict denominator);
