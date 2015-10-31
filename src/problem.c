
#include "problem.h"
#include <math.h>

void init_quadrature_weights(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // Create tempoary on host for quadrature weights
    double *quad_weights = malloc(sizeof(double)*problem->nang);
    // Uniform weights
    for (unsigned int a = 0; a < problem->nang; a++)
    {
        quad_weights[a] = 0.125 / (double)(problem->nang);
    }

    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->quad_weights, CL_TRUE,
        0, sizeof(double)*problem->nang, quad_weights, 0, NULL, NULL);
    check_ocl(err, "Copying quadrature weights to device");
    free(quad_weights);
}

void calculate_cosine_coefficients(const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers,
    double * restrict mu,
    double * restrict eta,
    double * restrict xi
    )
{

    double dm = 1.0 / problem->nang;

    mu[0] = 0.5 * dm;
    eta[0] = 1.0 - 0.5 * dm;
    double t = mu[0] * mu[0] + eta[0] * eta[0];
    xi[0] = sqrt(1.0 - t);

    for (unsigned int a = 1; a < problem->nang; a++)
    {
        mu[a] = mu[a-1] + dm;
        eta[a] = eta[a-1] - dm;
        t = mu[a] * mu[a] + eta[a] * eta[a];
        xi[a] = sqrt(1.0 - t);
    }

    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->mu, CL_FALSE,
        0, sizeof(double)*problem->nang, mu, 0, NULL, NULL);
    check_ocl(err, "Copying mu cosine to device");
    err = clEnqueueWriteBuffer(context->queue, buffers->eta, CL_FALSE,
        0, sizeof(double)*problem->nang, eta, 0, NULL, NULL);
    check_ocl(err, "Copying eta cosine to device");
    err = clEnqueueWriteBuffer(context->queue, buffers->xi, CL_TRUE,
        0, sizeof(double)*problem->nang, xi, 0, NULL, NULL);
    check_ocl(err, "Copying xi cosine to device");

}

void calculate_scattering_coefficients(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers,
    const double * restrict mu,
    const double * restrict eta,
    const double * restrict xi
    )
{
    // Allocate temporary on host for scattering coefficients
    double *scat_coeff = malloc(sizeof(double)*problem->nang*problem->cmom*8);
    // (mu*eta*xi)^l starting at 0
    for (int kd = 0; kd < 2; kd++)
    {
        double ks = (kd == 1) ? 1.0 : -1.0;
        for (int jd = 0; jd < 2; jd++)
        {
            double js = (jd == 1) ? 1.0 : -1.0;
            for (int id = 0; id < 2; id++)
            {
                double is = (id == 1) ? 1.0 : -1.0;
                int oct = 4*kd + 2*jd + id;
                // Init first moment
                for (unsigned int a = 0; a < problem->nang; a++)
                    scat_coeff[SCAT_COEFF_INDEX(a,0,oct,problem->nang,problem->cmom)] = 1.0;
                // Init other moments
                int mom = 1;
                for (int l = 1; l < problem->nmom; l++)
                {
                    for (int m = 0; m < 2*l+1; m++)
                    {
                        for (unsigned int a = 0; a < problem->nang; a++)
                        {
                            scat_coeff[SCAT_COEFF_INDEX(a,mom,oct,problem->nang,problem->cmom)] = pow(is*mu[a], 2.0*l-1.0) * pow(ks*xi[a]*js*eta[a], m);
                        }
                        mom += 1;
                    }
                }
            }
        }
    }

    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->scat_coeff, CL_TRUE,
        0, sizeof(double)*problem->nang*problem->cmom*8, scat_coeff, 0, NULL, NULL);
    check_ocl(err, "Copying scattering coefficients to device");
    free(scat_coeff);
}

void init_material_data(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers,
    double * restrict mat_cross_section
    )
{
    mat_cross_section[0] = 1.0;
    for (unsigned int g = 1; g < problem->ng; g++)
    {
        mat_cross_section[g] = mat_cross_section[g-1] + 0.01;
    }
    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->mat_cross_section, CL_TRUE,
        0, sizeof(double)*problem->ng, mat_cross_section, 0, NULL, NULL);
    check_ocl(err, "Copying material cross sections to device");
}

void init_fixed_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // Allocate temporary array for fixed source
    double *fixed_source = malloc(sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);

    // Source everywhere, set at strength 1.0
    // This is src_opt == 0 in original SNAP
    for(unsigned int k = 0; k < rankinfo->nz; k++)
        for(unsigned int j = 0; j < rankinfo->ny; j++)
            for(unsigned int i = 0; i < rankinfo->nx; i++)
                for(unsigned int g = 0; g < problem->ng; g++)
                    fixed_source[FIXED_SOURCE_INDEX(g,i,j,k,problem->ng,rankinfo->nx,rankinfo->ny)] = 1.0;

    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->fixed_source, CL_TRUE,
        0, sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz, fixed_source, 0, NULL, NULL);
    check_ocl(err, "Copying fixed source to device");
    free(fixed_source);
}

void init_scattering_matrix(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers,
    const double * restrict mat_cross_section
    )
{
    // Allocate temporary array for scattering matrix
    double *scattering_matrix = malloc(sizeof(double)*problem->nmom*problem->ng*problem->ng);

    // 10% up scattering
    // 20% in group scattering
    // 70% down scattering
    // First and last group, no up/down scattering
    for (unsigned int g = 0; g < problem->ng; g++)
    {
        if (problem->ng == 1)
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,problem->nmom,problem->ng)] = mat_cross_section[g] * 0.5;
            break;
        }

        scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g,problem->nmom,problem->ng)] = 0.2 * 0.5 * mat_cross_section[g];

        if (g > 0)
        {
            double t = 1.0 / (double)(g);
            for (unsigned int g2 = 0; g2 < g; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,problem->nmom,problem->ng)] = 0.1 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,problem->nmom,problem->ng)] = 0.3 * 0.5 * mat_cross_section[0];
        }

        if (g < (problem->ng) - 1)
        {
            double t = 1.0 / (double)(problem->ng - g - 1);
            for (unsigned int g2 = g + 1; g2 < problem->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,problem->nmom,problem->ng)] = 0.7 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,problem->ng-1,problem->ng-1,problem->nmom,problem->ng)] = 0.9 * 0.5 * mat_cross_section[problem->ng-1];
        }
    }

    // Set scattering moments (up to 4)
    // Second moment 10% of first, subsequent half of previous
    if (problem->nmom > 1)
    {
        for (unsigned int g1 = 0; g1 < problem->ng; g1++)
        {
            for (unsigned int g2 = 0; g2 < problem->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(1,g1,g2,problem->nmom,problem->ng)] = 0.1 * scattering_matrix[SCATTERING_MATRIX_INDEX(0,g1,g2,problem->nmom,problem->ng)];
                for (unsigned int m = 2; m < problem->nmom; m++)
                {
                    scattering_matrix[SCATTERING_MATRIX_INDEX(m,g1,g2,problem->nmom,problem->ng)] = 0.5 * scattering_matrix[SCATTERING_MATRIX_INDEX(m-1,g1,g2,problem->nmom,problem->ng)];
                }
            }
        }
    }

    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->scattering_matrix, CL_TRUE,
        0, sizeof(double)*problem->nmom*problem->ng*problem->ng, scattering_matrix, 0, NULL, NULL);
    check_ocl(err, "Copying scattering matrix to device");
    free(scattering_matrix);
}

void init_velocities(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // Allocate tempoary array for velocities
    double *velocities = malloc(sizeof(double)*problem->ng);

    for (unsigned int g = 0; g < problem->ng; g++)
        velocities[g] = (double)(problem->ng - g);

    // Copy to device
    cl_int err;
    err = clEnqueueWriteBuffer(context->queue, buffers->velocities, CL_TRUE,
        0, sizeof(double)*problem->ng, velocities, 0, NULL, NULL);
    check_ocl(err, "Copying velocities to device");
    free(velocities);
}

void init_velocity_delta(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    cl_int err;
    err = clSetKernelArg(context->kernels.calc_velocity_delta, 0, sizeof(cl_mem), &buffers->velocities);
    err |= clSetKernelArg(context->kernels.calc_velocity_delta, 1, sizeof(double), &problem->dt);
    err |= clSetKernelArg(context->kernels.calc_velocity_delta, 2, sizeof(cl_mem), &buffers->velocity_delta);
    check_ocl(err, "Setting velocity delta calculation kernel arguments");

    size_t global[] = {problem->ng};
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.calc_velocity_delta,
        1, 0, global, NULL,
        0, NULL, &velocity_delta_event);
    check_ocl(err, "Enqueue velocity delta calculation kernel");
}

void calculate_dd_coefficients(
    const struct problem * problem,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    cl_int err;
    err = clSetKernelArg(context->kernels.calc_dd_coeff, 0, sizeof(double), &problem->dx);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 1, sizeof(double), &problem->dy);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 2, sizeof(double), &problem->dz);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 3, sizeof(cl_mem), &buffers->eta);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 4, sizeof(cl_mem), &buffers->xi);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 5, sizeof(cl_mem), &buffers->dd_i);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 6, sizeof(cl_mem), &buffers->dd_j);
    err |= clSetKernelArg(context->kernels.calc_dd_coeff, 7, sizeof(cl_mem), &buffers->dd_k);
    check_ocl(err, "Setting diamond difference calculation kernel arguments");

    size_t global[] = {problem->nang};
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.calc_dd_coeff,
        1, 0, global, NULL,
        0, NULL, NULL);
    check_ocl(err, "Enqueue diamond difference calculation kernel");
}

void calculate_denominator(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    cl_int err;
    err = clSetKernelArg(context->kernels.calc_denominator, 0, sizeof(unsigned int), &rankinfo->nx);
    err |= clSetKernelArg(context->kernels.calc_denominator, 1, sizeof(unsigned int), &rankinfo->ny);
    err |= clSetKernelArg(context->kernels.calc_denominator, 2, sizeof(unsigned int), &rankinfo->nz);
    err |= clSetKernelArg(context->kernels.calc_denominator, 3, sizeof(unsigned int), &problem->nang);
    err |= clSetKernelArg(context->kernels.calc_denominator, 4, sizeof(unsigned int), &problem->ng);
    err |= clSetKernelArg(context->kernels.calc_denominator, 5, sizeof(cl_mem), &buffers->mat_cross_section);
    err |= clSetKernelArg(context->kernels.calc_denominator, 6, sizeof(cl_mem), &buffers->velocity_delta);
    err |= clSetKernelArg(context->kernels.calc_denominator, 7, sizeof(cl_mem), &buffers->mu);
    err |= clSetKernelArg(context->kernels.calc_denominator, 8, sizeof(cl_mem), &buffers->dd_i);
    err |= clSetKernelArg(context->kernels.calc_denominator, 9, sizeof(cl_mem), &buffers->dd_j);
    err |= clSetKernelArg(context->kernels.calc_denominator, 10, sizeof(cl_mem), &buffers->dd_k);
    err |= clSetKernelArg(context->kernels.calc_denominator, 11, sizeof(cl_mem), &buffers->denominator);
    check_ocl(err, "Setting denominator kernel arguments");

    size_t global[] = {problem->nang, problem->ng};
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.calc_denominator,
        2, 0, global, NULL,
        0, NULL, &denominator_event);
    check_ocl(err, "Enqueue denominator kernel");
}
