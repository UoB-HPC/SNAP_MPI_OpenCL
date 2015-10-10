
#pragma OPENCL EXTENSION cl_khr_fp64 : enable


#define SOURCE_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+((cmom)*(g))+((cmom)*(ng)*(i))+((cmom)*(ng)*(nx)*(j))+((cmom)*(ng)*(nx)*(ny)*(k)))
#define FIXED_SOURCE_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))
#define SCATTERING_MATRIX_INDEX(m,g1,g2,nmom,ng) ((m)+((nmom)*(g1))+((nmom)*(ng)*(g2)))
#define SCALAR_FLUX_INDEX(g,i,j,k,ng,nx,ny) ((g)+((ng)*(i))+((ng)*(nx)*(j))+((ng)*(nx)*(ny)*(k)))
#define SCALAR_FLUX_MOMENTS_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+(((cmom)-1)*(g))+(((cmom)-1)*(ng)*(i))+(((cmom)-1)*(ng)*(nx)*(j))+(((cmom)-1)*(ng)*(nx)*(ny)*(k)))


#define outer_source(m,g,i,j,k) outer_source[SOURCE_INDEX((m),(g),(i),(j),(k),cmom,ng,nx,ny)]
#define fixed_source(g,i,j,k) fixed_source[FIXED_SOURCE_INDEX((g),(i),(j),(k),ng,nx,ny)]
#define scattering_matrix(m,g1,g2) scattering_matrix[SCATTERING_MATRIX_INDEX((m),(g1),(g2),nmom,ng)]
#define scalar_flux(g,i,j,k) scalar_flux[SCALAR_FLUX_INDEX((g),(i),(j),(k),ng,nx,ny)]
#define scalar_flux_moments(m,g,i,j,k) scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX((m),(g),(i),(j),(k),cmom,ng,nx,ny)]

// 3D kernel, in local nx,ny,nz dimensions
// Probably not going to vectorise very well..
kernel void calc_outer_source(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int cmom,
    const unsigned int nmom,
    global const double * restrict fixed_source,
    global const double * restrict scattering_matrix,
    global const double * restrict scalar_flux,
    global const double * restrict scalar_flux_moments,
    global double * restrict outer_source
    )
{
    const size_t i = get_global_id(0);
    const size_t j = get_global_id(1);
    const size_t k = get_global_id(2);

    for (unsigned int g = 0; g < ng; g++)
    {
        // Set first moment to the fixed source
        outer_source(0,g,i,j,k) = fixed_source(g,i,j,k);

        // Loop over groups and moments to compute out-of-group scattering
        for (unsigned int g2 = 0; g < ng; g2++)
        {
            if (g == g2)
                continue;
            // Compute scattering source
            outer_source(0,g,i,j,k) += scattering_matrix(0,g2,g) * scalar_flux(g2,i,j,k);
            // Other moments
            unsigned int mom = 1;
            for (unsigned int l = 0; l < nmom; l++)
            {
                for (unsigned int m = 0; m < 2*l+1; m++)
                {
                    outer_source(mom,g,i,j,k) += scattering_matrix(l,g2,g) * scalar_flux_moments(mom-1,g2,i,j,k);
                    mom += 1;
                }
            }
        }
    }

}
