
#include <stdlib.h>

#include "problem.h"
#include "allocate.h"

void allocate_memory(struct problem globals, struct rankinfo local, struct mem * memory)
{
	// Allocate two copies of the angular flux
	// grid * angles * noct (8) * ng
	memory->angular_flux_in = malloc(sizeof(double)*local.nx*local.ny*local.nz*globals.nang*8*globals.ng);
	memory->angular_flux_out = malloc(sizeof(double)*local.nx*local.ny*local.nz*globals.nang*8*globals.ng);

	// Allocate edge arrays
	memory->flux_i = malloc(sizeof(double)*globals.nang*globals.ng*local.ny*local.nz);
	memory->flux_j = malloc(sizeof(double)*globals.nang*globals.ng*local.nx*local.nz);
	memory->flux_k = malloc(sizeof(double)*globals.nang*globals.ng*local.nx*local.ny);

	// Scalar flux
	// grid * ng
	memory->scalar_flux_in = malloc(sizeof(double)*local.nx*local.ny*local.nz*globals.ng);
	memory->scalar_flux_out = malloc(sizeof(double)*local.nx*local.ny*local.nz*globals.ng);

	// Quadrature weights
	memory->quad_weights = malloc(sizeof(double)*globals.nang);

	// Cosine coefficients
	memory->mu = malloc(sizeof(double)*globals.nang);
	memory->eta = malloc(sizeof(double)*globals.nang);
	memory->xi = malloc(sizeof(double)*globals.nang);

	// Scattering coefficient
	memory->scat_coeff = malloc(sizeof(double)*globals.nang*globals.cmom*8);

	// Material cross section
	memory->mat_cross_section = malloc(sizeof(double)*globals.ng);

	// Fixed source
	memory->fixed_source = malloc(sizeof(double)*globals.ng*local.nx*local.ny*local.nz);

	// Scattering matrix
	memory->scattering_matrix = malloc(sizeof(double)*globals.nmom*globals.ng*globals.ng);

	// Diamon difference co-efficients
	memory->dd_i = malloc(sizeof(double));
	memory->dd_j = malloc(sizeof(double)*globals.nang);
	memory->dd_k = malloc(sizeof(double)*globals.nang);

	// Mock velocities array
	memory->velocities = malloc(sizeof(double)*globals.ng);

}

void free_memory(struct mem * memory)
{
	free(memory->angular_flux_in);
	free(memory->angular_flux_out);
	free(memory->flux_i);
	free(memory->flux_j);
	free(memory->flux_k);
	free(memory->scalar_flux_in);
	free(memory->scalar_flux_out);
	free(memory->mu);
	free(memory->eta);
	free(memory->xi);
}
