
#pragma once

#include "ocl_global.h"

struct buffers
{
    // Angular flux - two copies for time dependence, each ocant in own buffer
    cl_mem angular_flux_in[8];
    cl_mem angular_flux_out[8];

    // Edge flux arrays
    cl_mem flux_i, flux_j, flux_k;

    // Scalar flux arrays
    cl_mem scalar_flux;
    cl_mem scalar_flux_moments;

    // Quadrature weights
    cl_mem quad_weights;

    // Cosine coefficients
    cl_mem mu, eta, xi;

    // Scattering coefficient
    cl_mem scat_coeff;

    // Material cross section
    cl_mem mat_cross_section;

    // Source terms
    cl_mem fixed_source;
    cl_mem outer_source;
    cl_mem inner_source;

    // Scattering terms
    cl_mem scattering_matrix;

    // Diamond diference co-efficients
    cl_mem dd_i, dd_j, dd_k;

    // Mock velocities array
    cl_mem velocities;
    cl_mem velocity_delta;

    // Transport denominator
    cl_mem denominator;
};
