
/** \file
* \brief Calculate "static" problem data
*
* Routines to initilise the data arrays based on the problem inputs.
* Also contains data calculated every outer, which could only be done once
* per timestep.
*/

#pragma once

#include "global.h"
#include "ocl_global.h"
#include "ocl_buffers.h"

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

/** \brief Index for scalar flux array */
#define SCALAR_FLUX_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))

/** \brief Index for scalar flux moments array */
#define SCALAR_FLUX_MOMENTS_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+(((cmom)-1)*(g))+(((cmom)-1)*(ng)*(i))+(((cmom)-1)*(ng)*(nx)*(j))+(((cmom)-1)*(ng)*(nx)*(ny)*(k)))

/**@}*/


/** \brief Initilise quadrature weights
*
* Set to uniform weights: number of angles divided by eight.
*/
void init_quadrature_weights(const struct problem * problem, const struct context * context, const struct buffers * buffers);

/** \brief Calculate cosine coefficients
*
* Populates the \a mu, \a eta and \a xi arrays.
*/
void calculate_cosine_coefficients(const struct problem * problem, const struct context * context, const struct buffers * buffers, double * restrict mu, double * restrict eta, double * restrict xi);

/** \brief Calculate the scattering coefficients
*
* Populates the \a scat_coef array based on the cosine coefficients.
* Set as \f$(\mu*\eta*\xi)^l\f$ starting at 0, for the lth moment.
*/
void calculate_scattering_coefficients(const struct problem * problem, const struct context * context, const struct buffers * buffers, const double * restrict mu, const double * restrict eta, const double * restrict xi);

/** \brief Set material cross sections
*
* We one have one material across the whole grid. Set to 1.0 for the first group, and + 0.01 for each subsequent group.
*/
void init_material_data(const struct problem * problem, const struct context * context, const struct buffers * buffers,double * restrict mat_cross_section);

/** /brief Set fixed source data
*
* Source is applied everywhere, set at strenght 1.0.
* This is fixed src_opt == 0 in original SNAP
*/
void init_fixed_source(const struct problem * problem, const struct rankinfo * rankinfo, const struct context * context, const struct buffers * buffers);

/** \brief Setup group to group scattering information
*
* Scattering is 10% upscattering, 20% in group and 70% down scattering in every group,
* except first and last which have no up/down scattering.
* Data is initilised for all moments.
*/
void init_scattering_matrix(const struct problem * problem, const struct context * context, const struct buffers * buffers, const double * restrict mat_cross_section);


/** \brief Set velocities array
*
* Fake data on group velocity.
*/
void init_velocities(const struct problem * problem, double * restrict velocities);

/** \brief Set velocity time delta array */
void init_velocity_delta(const struct problem * problem, const double * restrict velocities, double * restrict velocity_delta);

/** \brief Calculate the spatial diamond difference coefficients
*
* Called every outer. Includes the cosine coefficient terms.
*/
void calculate_dd_coefficients(const struct problem * problem, const double * restrict eta, const double * restrict xi, double * restrict dd_i, double * restrict dd_j, double * restrict dd_k);

/** \brief Calculate the denominator to the transport equation update
*
* Called every outer.
*/
void calculate_denominator(const struct problem * problem, const struct rankinfo * rankinfo, const double * restrict dd_i, const double * restrict dd_j, const double * restrict dd_k, const double * restrict mu, const double * restrict mat_cross_section, const double * restrict velocity_delta, double * restrict denominator);
