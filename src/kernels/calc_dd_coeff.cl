
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

kernel void calc_dd_coeff(
    const double dx,
    const double dy,
    const double dz,
    global const double * restrict eta,
    global const double * restrict xi,
    global double * restrict dd_i,
    global double * restrict dd_j,
    global double * restrict dd_k
    )
{
    size_t a = get_global_id(0);

    // There is only one dd_i so just get the first work-item to do this
    if (a == 0 && get_group_id(0) == 0)
        dd_i[0] = 2.0 / dx;

    dd_j[a] = (2.0 / dy) * eta[a];
    dd_k[a] = (2.0 / dz) * xi[a];
}
