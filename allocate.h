
#pragma once

#include "comms.h"

// Struct to hold the buffers
struct mem
{
	// Angular flux
	double *angular_flux_in;
	double *angular_flux_out;

	// Edge flux arrays
	double *flux_i;
	double *flux_j;
	double *flux_k;

	// Scalar flux
	double *scalar_flux_in;
	double *scalar_flux_out;

	// Quadrature weights
	double *quad_weights;

	// Cosine coefficients
	double *mu;
	double *eta;
	double *xi;

	// Scattering coefficient
	double *scat_coeff;

	// Material cross sections - ASSUME ONE MATERIAL
	double *mat_cross_section;

	// Fixed source
	double *fixed_source;

	// Scattering matrix
	double *scattering_matrix;

	// Diamond difference co-efficients
	double *dd_i;
	double *dd_j;
	double *dd_k;

	// Mock velocities array
	double *velocities;

};

void allocate_memory(struct problem globals, struct rankinfo local, struct mem * memory);
void free_memory(struct mem * memory);

