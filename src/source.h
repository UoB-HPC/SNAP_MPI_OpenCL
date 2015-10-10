
#pragma once

/** \file
* \brief Source update routines
*/

#include "global.h"
#include "problem.h"

#include "ocl_global.h"
#include "ocl_buffers.h"

/** \ingroup MEM
* @{
* \brief Index for source arrays */
#define SOURCE_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+((cmom)*(g))+((cmom)*(ng)*(i))+((cmom)*(ng)*(nx)*(j))+((cmom)*(ng)*(nx)*(ny)*(k)))
/**@}*/

/** \brief Compute the outer source on the device (non-blocking)
*
* First moment is set to fixed source. Subsequent momemnts
* use group-to-group scattering.
*/
void compute_outer_source(const struct problem * problem, const struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);

/** \brief Computer the inner source
*
* Set to the outer source plus within group scattering based on scalar flux and scalar flux moments.
*/
void compute_inner_source(
    const struct problem * global,
    const struct rankinfo * rankinfo,
    const double * restrict outer_source,
    const double * restrict scattering_matrix,
    const double * restrict scalar_flux,
    const double * restrict scalar_flux_moments,
    double * restrict inner_source
    );
