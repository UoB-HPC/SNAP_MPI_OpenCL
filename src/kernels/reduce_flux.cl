
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// We want to perform a weighted sum of angles in each cell in each energy group
// One work-group per cell per energy group, and reduce within a work-group
kernel void reduce_flux(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,

    global const double * restrict angular_flux_in_0,
    global const double * restrict angular_flux_in_1,
    global const double * restrict angular_flux_in_2,
    global const double * restrict angular_flux_in_3,
    global const double * restrict angular_flux_in_4,
    global const double * restrict angular_flux_in_5,
    global const double * restrict angular_flux_in_6,
    global const double * restrict angular_flux_in_7,

    global const double * restrict angular_flux_out_0,
    global const double * restrict angular_flux_out_1,
    global const double * restrict angular_flux_out_2,
    global const double * restrict angular_flux_out_3,
    global const double * restrict angular_flux_out_4,
    global const double * restrict angular_flux_out_5,
    global const double * restrict angular_flux_out_6,
    global const double * restrict angular_flux_out_7,

    global const double * restrict velocity_delta,

    global double * restrict scalar_flux,
    local double * restrict local_scalar
    )
{
    ;
}