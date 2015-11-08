
#pragma once

/** \file
* \brief Contains list of kernel string files
*/

#include "kernels/calc_velocity_delta.h"
#include "kernels/calc_dd_coeff.h"
#include "kernels/calc_denominator.h"
#include "kernels/zero_buffer.h"
#include "kernels/outer_source.h"
#include "kernels/inner_source.h"
#include "kernels/sweep_plane.h"
#include "kernels/reduce_flux.h"
#include "kernels/reduce_flux_moments.h"

/** \brief Array of OpenCL kernel strings */
const char * ocl_kernels[] = {
    calc_velocity_delta_ocl,
    calc_dd_coeff_ocl,
    calc_denominator_ocl,
    zero_buffer_ocl,
    outer_source_ocl,
    inner_source_ocl,
    sweep_plane_ocl,
    reduce_flux_ocl,
    reduce_flux_moments_ocl
};
