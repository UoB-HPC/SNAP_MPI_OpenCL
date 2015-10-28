
/** \file
* \brief Manage the allocation of arrays
*
* All problem scope arrays are allocated in the host DRAM using these functions calls.
*/

#pragma once

#include "global.h"

/** \brief Struct to hold the buffers
*
* All the memory arrays are stored here
*/
struct memory
{
	/**@{ \brief Angular flux */
	/**
	* Size: (nang, ng, nx, ny, nz, 8)
	*
	* Note, rankinfo spatial dimensions, 8 octants
	*/
	double *angular_flux_in;
	double *angular_flux_out;
	/**@}*/

	/**@{*/
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, ny, nz)
	*
	* Note, rankinfo spatial dimension
	*/
	double *flux_i;
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, nx, nz)
	*
	* Note, rankinfo spatial dimension
	*/
	double *flux_j;
	/** \brief Edge flux arrays */
	/** Size: (nang, ng, nx, ny)
	*
	* Note, rankinfo spatial dimension
	*/
	double *flux_k;
	/**@}*/

	/**@{ \brief Scalar flux */
	/**
	* Size: (ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimensions
	*/
	double *scalar_flux;
	double *old_inner_scalar_flux;
	double *old_outer_scalar_flux;
	/**@}*/

	/**@{*/
	/** \brief Scalar flux moments */
	/** Size: (cmom-1, ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimensions
	*/
	double *scalar_flux_moments;
	/**@}*/

	/**@{ \brief Cosine coefficients */
	/** Size: (nang) */
	double *mu;
	double *eta;
	double *xi;
	/**@}*/

	/** \brief Material cross sections
	*
	* ASSUME ONE MATERIAL
	*
	* Size: (ng)
	*/
	double *mat_cross_section;


};

/** \brief Allocate the problem arrays */
void allocate_memory(struct problem * problem, struct rankinfo * rankinfo, struct memory * memory);

/** \brief Free the arrays sroted in the \a mem struct */
void free_memory(struct memory * memory);

