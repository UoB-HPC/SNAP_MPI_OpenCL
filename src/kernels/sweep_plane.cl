
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

struct cell_id
{
    unsigned int i, j, k;
};


#define SOURCE_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+((cmom)*(g))+((cmom)*(ng)*(i))+((cmom)*(ng)*(nx)*(j))+((cmom)*(ng)*(nx)*(ny)*(k)))
#define SCAT_COEFF_INDEX(a,l,o,nang,cmom) ((a)+((nang)*(l))+((nang)*(cmom)*o))
#define FLUX_I_INDEX(a,g,j,k,nang,ng,ny) ((a)+((nang)*(g))+((nang)*(ng)*(j))+((nang)*(ng)*(ny)*(k)))
#define FLUX_J_INDEX(a,g,i,k,nang,ng,nx) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(k)))
#define FLUX_K_INDEX(a,g,i,j,nang,ng,nx) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j)))
#define ANGULAR_FLUX_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define DENOMINATOR_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))

#define source(m,g,i,j,k) source[SOURCE_INDEX((m),(g),(i),(j),(k),cmom,ng,nx,ny)]
#define scat_coeff(a,l,o) scat_coeff[SCAT_COEFF_INDEX((a),(l),(o),nang,cmom)]
#define flux_i(a,g,j,k) flux_i[FLUX_I_INDEX((a),(g),(j),(k),nang,ng,ny)]
#define flux_j(a,g,i,k) flux_j[FLUX_J_INDEX((a),(g),(i),(k),nang,ng,nx)]
#define flux_k(a,g,i,j) flux_k[FLUX_K_INDEX((a),(g),(i),(j),nang,ng,nx)]
#define angular_flux_in(a,g,i,j,k) angular_flux_in[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out(a,g,i,j,k) angular_flux_out[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define denominator(a,g,i,j,k) denominator[DENOMINATOR_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]


kernel void sweep_plane(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int cmom,
    const int istep,
    const int jstep,
    const int kstep,
    const unsigned int oct,
    global const struct cell_id * plane,
    global const double * restrict source,
    global const double * restrict scat_coeff,
    global const double * restrict dd_i,
    global const double * restrict dd_j,
    global const double * restrict dd_k,
    global const double * restrict mu,
    global const double * restrict velocity_delta,
    global const double * restrict mat_cross_section,
    global const double * restrict denominator,
    global const double * restrict angular_flux_in,
    global double * restrict flux_i,
    global double * restrict flux_j,
    global double * restrict flux_k,
    global double * restrict angular_flux_out
    )
{
    // Recover indexes for angle and group
    const size_t a = get_global_id(0) % nang;
    const size_t g = get_global_id(0) / nang;

    // Read cell index from plane buffer
    const size_t i = (istep > 0) ? plane[get_global_id(1)].i : nx - plane[get_global_id(1)].i - 1;
    const size_t j = (jstep > 0) ? plane[get_global_id(1)].j : ny - plane[get_global_id(1)].j - 1;
    const size_t k = (kstep > 0) ? plane[get_global_id(1)].k : nz - plane[get_global_id(1)].k - 1;

    //
    // Compute the angular flux (psi)
    //

    // Begin with the first scattering moment
    double source_term = source(0,g,i,j,k);

    // Add in the anisotropic scattering source moments
    for (unsigned int l = 1; l < cmom; l++)
    {
        source_term += scat_coeff(a,l,oct) * source(l,g,i,j,k);
    }

    double psi =
        source_term
        + flux_i(a,g,j,k)*mu[a]*dd_i[0]
        + flux_j(a,g,i,k)*dd_j[a]
        + flux_k(a,g,i,j)*dd_k[a];

    // Add contribution from last timestep flux if time-dependant
    if (velocity_delta[g] != 0.0)
    {
        psi += velocity_delta[g] * angular_flux_in(a,g,i,j,k);
    }

    // "Divide" by denominator
    psi *= denominator(a,g,i,j,k);

    // Compute upwind fluxes
    double tmp_flux_i = 2.0 * psi - flux_i(a,g,j,k);
    double tmp_flux_j = 2.0 * psi - flux_j(a,g,i,k);
    double tmp_flux_k = 2.0 * psi - flux_k(a,g,i,j);

    // Time difference the final flux value
    if (velocity_delta[g] != 0.0)
    {
        psi = 2.0 * psi - angular_flux_in(a,g,i,j,k);
    }

    // Fixup
    double zeros[4];
    int num_ok = 4;
    for (int fix = 0; fix < 4; fix++)
    {
        zeros[0] = (tmp_flux_i < 0.0) ? 0.0 : 1.0;
        zeros[1] = (tmp_flux_j < 0.0) ? 0.0 : 1.0;
        zeros[2] = (tmp_flux_k < 0.0) ? 0.0 : 1.0;
        zeros[3] = (psi < 0.0)        ? 0.0 : 1.0;

        if (num_ok == zeros[0] + zeros[1] + zeros[2] + zeros[3])
            continue;

        num_ok = zeros[0] + zeros[1] + zeros[2] + zeros[3];

        // Recalculate psi
        psi =
            flux_i(a,g,j,k)*mu[a]*dd_i[0]*(1.0 + zeros[0]) +
            flux_j(a,g,i,k)*dd_j[a]*(1.0 + zeros[1]) +
            flux_k(a,g,i,j)*dd_k[a]*(1.0 + zeros[2]);

        if (velocity_delta[g] != 0.0)
        {
            psi += velocity_delta[g] * angular_flux_in(a,g,i,j,k) * (1.0 + zeros[3]);
        }

        psi = 0.5 * psi + source_term;

        double new_denominator =
            mat_cross_section[g] +
            mu[a] * dd_i[0] * zeros[0] +
            dd_j[a] * zeros[1] +
            dd_k[a] * zeros[2] +
            velocity_delta[g] * zeros[3];
        if (new_denominator > 1.0E-12)
        {
            psi /= new_denominator;
        }
        else
        {
            psi = 0.0;
        }

        tmp_flux_i = 2.0 * psi - flux_i(a,g,j,k);
        tmp_flux_j = 2.0 * psi - flux_j(a,g,i,k);
        tmp_flux_k = 2.0 * psi - flux_k(a,g,i,j);

        if (velocity_delta[g] != 0.0)
        {
            psi = 2.0 * psi - angular_flux_in(a,g,i,j,k);
        }

    }

    // Write values to global memory
    flux_i(a,g,j,k) = tmp_flux_i * zeros[0];
    flux_j(a,g,i,k) = tmp_flux_j * zeros[1];
    flux_k(a,g,i,j) = tmp_flux_k * zeros[2];
    angular_flux_out(a,g,i,j,k) = psi * zeros[3];
}
