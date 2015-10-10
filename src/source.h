
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

/** \brief Compute the inner source on the device (non-blocking)
*
* Set to the outer source plus within group scattering based on scalar flux and scalar flux moments.
*/
void compute_inner_source(const struct problem * problem, const struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);
