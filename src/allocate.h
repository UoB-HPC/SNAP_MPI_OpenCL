
/** \file
* \brief Manage the allocation of arrays
*
* All problem scope arrays are allocated in the host DRAM using these functions calls.
*/

#pragma once

#include "comms.h"

/** \brief Struct to hold the buffers
*
* All the memory arrays are stored here
*/
struct mem
{
	/**@{ \brief Angular flux */
	/**
	* Size: (nang, ng, nx, ny, nz, 8)
	*
	* Note, local spatial dimensions, 8 octants
	*/
	double *angular_flux_in;
	double *angular_flux_out;
	/**@}*/

	/**@{*/
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, ny, nz)
	*
	* Note, local spatial dimension
	*/
	double *flux_i;
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, nx, nz)
	*
	* Note, local spatial dimension
	*/
	double *flux_j;
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, nx, ny)
	*
	* Note, local spatial dimension
	*/
	double *flux_k;
	/**@}*/

	/**@{ \brief Scalar flux */
	/**
	* Size: (ng, nx, ny, nz)
	*
	* Note, local spatial dimensions
	*/
	double *scalar_flux_in;
	double *scalar_flux_out;
	/**@}*/

	/** \brief Quadrature weights */
	/** Size: (ng) */
	double *quad_weights;

	/**@{ \brief Cosine coefficients */
	/** Size: (ng) */
	double *mu;
	double *eta;
	double *xi;
	/**@}*/

	/** \brief Scattering coefficient */
	/** Size: (nang, cmom, octant) */
	double *scat_coeff;

	/** \brief Material cross sections
	*
	* ASSUME ONE MATERIAL
	*
	* Size: (ng)
	*/
	double *mat_cross_section;

	/**@{*/
	/** \brief Fixed source */
	/** Size: (ng, nx, ny, nz)
	*
	* Note, local spatial dimension
	*/
	double *fixed_source;
	/** \brief Outer source: group-to-group scattering plus fixed source */
	/** Size: (cmom, ng, nx, ny, nz)
	*
	* Note, local spatial dimension
	*/
	double *outer_source;
	/** \brief Inner (total) source: outer source plus with-group source */
	/** Size: (cmom, ng, nx, ny, nz)
	*
	* Note, local spatial dimension
	*/
	double *inner_source;
	/**@}*/

	/** \brief Scattering matrix */
	/** Size: (nmom, ng, ng) */
	double *scattering_matrix;

	/**@{*/
	/** \brief Diamond difference co-efficients */
	/** Size: 1 */
	double *dd_i;
	/** \brief Diamond difference co-efficients */
	/** Size: (nang) */
	double *dd_j;
	/** \brief Diamond difference co-efficients */
	/** Size: (nang) */
	double *dd_k;
	/**@}*/

	/** \brief Mock velocities array */
	/** Size: (ng) */
	double *velocities;

	/** \brief Time absorption coefficient */
	/** Size: (ng) */
	double *velocity_delta;

	/** \brief Denominator array */
	/**
	* Size: (nang, ng, nx, ny, nz)
	*
	* Note, local spatial dimensions
	*/
	double *denominator;

};

/** \brief Allocate the problem arrays */
void allocate_memory(struct problem globals, struct rankinfo local, struct mem * memory);

/** \brief Free the arrays sroted in the \a mem struct */
void free_memory(struct mem * memory);

