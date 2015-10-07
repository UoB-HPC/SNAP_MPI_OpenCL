
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Calculate the time absorbtion coefficient
kernel void calc_velocity_delta(
    global const double * restrict velocities,
    const double dt,
    global double * restrict velocity_delta
    )
{
    size_t g = get_global_id(0);
    velocity_delta[g] = 2.0 / (dt * velocities[g]);

}
