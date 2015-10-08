
#pragma once

#include "kernels/calc_velocity_delta.h"
#include "kernels/calc_dd_coeff.h"

const char * ocl_kernels[] = {
    calc_velocity_delta_ocl,
    calc_dd_coeff_ocl
};
