
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


#define ANGULAR_FLUX_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define SCALAR_FLUX_MOMENTS_INDEX(m,g,i,j,k,cmom_len,ng,nx,ny) ((m)+((cmom_len)*(g))+((cmom_len)*(ng)*(i))+((cmom_len)*(ng)*(nx)*(j))+((cmom_len)*(ng)*(nx)*(ny)*(k)))
#define SCAT_COEFF_INDEX(a,l,o,nang,cmom) ((a)+((nang)*(l))+((nang)*(cmom)*o))

#define angular_flux_in_0(a,g,i,j,k) angular_flux_in_0[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_1(a,g,i,j,k) angular_flux_in_1[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_2(a,g,i,j,k) angular_flux_in_2[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_3(a,g,i,j,k) angular_flux_in_3[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_4(a,g,i,j,k) angular_flux_in_4[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_5(a,g,i,j,k) angular_flux_in_5[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_6(a,g,i,j,k) angular_flux_in_6[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_7(a,g,i,j,k) angular_flux_in_7[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_0(a,g,i,j,k) angular_flux_out_0[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_1(a,g,i,j,k) angular_flux_out_1[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_2(a,g,i,j,k) angular_flux_out_2[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_3(a,g,i,j,k) angular_flux_out_3[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_4(a,g,i,j,k) angular_flux_out_4[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_5(a,g,i,j,k) angular_flux_out_5[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_6(a,g,i,j,k) angular_flux_out_6[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_7(a,g,i,j,k) angular_flux_out_7[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]

#define scalar_flux_moments(l,g,i,j,k) scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX((l),(g),(i),(j),(k),cmom_len,ng,nx,ny)]
#define scat_coeff(a,l,o) scat_coeff[SCAT_COEFF_INDEX((a),(l),(o),nang,cmom)]


// We want to perform a weighted sum of angles in each cell in each energy group for each moment
// One work-group per cell per energy group, and reduce within a work-group
// Work-groups must be power of two sized
kernel void reduce_flux_moments(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int cmom,

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
    global const double * restrict quad_weights,
    global const double * restrict scat_coeff,

    global double * restrict scalar_flux_moments,
    local double * restrict local_scalar
    )
{
    // Make sure cmom_len is cmom-1, but such that we index the scalar flux moments array properly
    const unsigned int cmom_len = (cmom-1 == 0) ? 1 : cmom-1;

    const size_t a = get_local_id(0);
    const size_t g = get_group_id(0);

    const size_t i = get_global_id(1) % nx;
    const size_t j = (get_global_id(1) / nx) % ny;
    const size_t k = get_global_id(1) / (nx * ny);

    for (unsigned int l = 0; l < cmom-1; l++)
    {
        // Load into local memory
        local_scalar[a] = 0.0;
        for (unsigned int aa = a; aa < nang; aa += get_local_size(0))
        {
            const double w = quad_weights[aa];
            if (velocity_delta[g] != 0.0)
            {
                local_scalar[a] +=
                    scat_coeff(aa,l+1,0) * w * (0.5 * (angular_flux_out_0(aa,g,i,j,k) + angular_flux_in_0(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,1) * w * (0.5 * (angular_flux_out_1(aa,g,i,j,k) + angular_flux_in_1(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,2) * w * (0.5 * (angular_flux_out_2(aa,g,i,j,k) + angular_flux_in_2(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,3) * w * (0.5 * (angular_flux_out_3(aa,g,i,j,k) + angular_flux_in_3(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,4) * w * (0.5 * (angular_flux_out_4(aa,g,i,j,k) + angular_flux_in_4(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,5) * w * (0.5 * (angular_flux_out_5(aa,g,i,j,k) + angular_flux_in_5(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,6) * w * (0.5 * (angular_flux_out_6(aa,g,i,j,k) + angular_flux_in_6(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,7) * w * (0.5 * (angular_flux_out_7(aa,g,i,j,k) + angular_flux_in_7(aa,g,i,j,k)));
            }
            else
            {
                local_scalar[a] +=
                    scat_coeff(aa,l+1,0) * w * angular_flux_out_0(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,1) * w * angular_flux_out_1(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,2) * w * angular_flux_out_2(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,3) * w * angular_flux_out_3(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,4) * w * angular_flux_out_4(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,5) * w * angular_flux_out_5(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,6) * w * angular_flux_out_6(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,7) * w * angular_flux_out_7(aa,g,i,j,k);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce in local memory
        for (unsigned int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
        {
            if (a < offset)
            {
                local_scalar[a] += local_scalar[a + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Save result
        if (a == 0)
        {
            scalar_flux_moments(l,g,i,j,k) = local_scalar[0];
        }
    } // End of moment loop
}
