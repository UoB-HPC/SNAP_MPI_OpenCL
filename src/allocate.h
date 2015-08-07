
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
	double *angular_flux_in;
	double *angular_flux_out;
	/**@}*/

	/**@{ \brief Edge flux arrays */
	double *flux_i;
	double *flux_j;
	double *flux_k;
	/**@}*/

	/**@{ \brief Scalar flux */
	double *scalar_flux_in;
	double *scalar_flux_out;
	/**@}*/

	/** \brief Quadrature weights */
	double *quad_weights;

	/**@{ \brief Cosine coefficients */
	double *mu;
	double *eta;
	double *xi;
	/**@}*/

	/** \brief Scattering coefficient */
	double *scat_coeff;

	/** \brief Material cross sections
	*
	* ASSUME ONE MATERIAL
	*/
	double *mat_cross_section;

	/**@{
	/** \brief Fixed source */
	double *fixed_source;
	/** \brief Outer source: group-to-group scattering plus fixed source */
	double *outer_source;
	/** \brief Inner (total) source: outer source plus with-group source */
	double *inner_source;
	/**@}*/

	/** \brief Scattering matrix */
	double *scattering_matrix;

	/**@{ \brief Diamond difference co-efficients */
	double *dd_i;
	double *dd_j;
	double *dd_k;
	/**@}*/

	/** \brief Mock velocities array */
	double *velocities;

	/** \brief Time absorption coefficient */
	double *velocity_delta;

	/** \brief Denominator array */
	double *denominator;

};

/** \brief Allocate the problem arrays */
void allocate_memory(struct problem globals, struct rankinfo local, struct mem * memory);

/** \brief Free the arrays sroted in the \a mem struct */
void free_memory(struct mem * memory);

