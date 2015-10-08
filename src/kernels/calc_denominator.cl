
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define DENOMINATOR_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define denominator(a,g,i,j,k) denominator[DENOMINATOR_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]

kernel void calc_denominator(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    global const double * restrict mat_cross_section,
    global const double * restrict velocity_delta,
    global const double * restrict mu,
    global const double * restrict dd_i,
    global const double * restrict dd_j,
    global const double * restrict dd_k,
    global double * restrict denominator
    )
{
    size_t a = get_global_id(0);
    size_t g = get_global_id(1);

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                denominator(a,g,i,j,k) = 1.0 / (mat_cross_section[g] + velocity_delta[g] + mu[a]*dd_i[0] + dd_j[a] + dd_k[a]);
}
