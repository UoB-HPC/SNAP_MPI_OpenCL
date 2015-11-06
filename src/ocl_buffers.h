
/** \file
* \brief Manage the allocation of OpenCL buffers
*/

#pragma once

#include "ocl_global.h"
#include "global.h"

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

    // Time absorption coefficient
    cl_mem velocity_delta;

    // Transport denominator
    cl_mem denominator;

    // Lists of cell indicies in each plane
    // Each buffer is an array of the i,j,k indicies for cells within that plane
    // One buffer per plane
    cl_mem *planes;
};

/** \brief Check the device has enough memory to allocate the buffers */
void check_device_memory_requirements(struct problem * problem, struct rankinfo * rankinfo, struct context * context);

/** \brief Allocate global device memory */
void allocate_buffers(struct problem * problem, struct rankinfo * rankinfo, struct context * context, struct buffers * buffers);

/** \brief Launch a kernel to zero the (1D) buffer (non-blocking) */
void zero_buffer(struct context * context, cl_mem buffer, size_t offset, size_t size);

/** \brief Swap the angular flux pointers around (in <-> out) */
void swap_angular_flux_buffers(struct buffers * buffers);
