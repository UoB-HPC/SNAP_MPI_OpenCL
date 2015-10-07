
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
	double *scalar_flux_in;
	double *scalar_flux_out;
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

	/**@{*/
	/** \brief Outer source: group-to-group scattering plus fixed source */
	/** Size: (cmom, ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimension
	*/
	double *outer_source;
	/** \brief Inner (total) source: outer source plus with-group source */
	/** Size: (cmom, ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimension
	*/
	double *inner_source;
	/**@}*/

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

	/** \brief Time absorption coefficient */
	/** Size: (ng) */
	double *velocity_delta;

	/** \brief Denominator array */
	/**
	* Size: (nang, ng, nx, ny, nz)
	*
	* Note, rankinfo spatial dimensions
	*/
	double *denominator;

};

/** \brief Allocate the problem arrays */
void allocate_memory(struct problem * problem, struct rankinfo * rankinfo, struct memory * memory);

/** \brief Free the arrays sroted in the \a mem struct */
void free_memory(struct memory * memory);

