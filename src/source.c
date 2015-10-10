
#include "source.h"


void compute_outer_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    struct context * context,
    struct buffers * buffers
    )
{
    cl_int err;
    err = clSetKernelArg(context->kernels.outer_source, 0, sizeof(unsigned int), &rankinfo->nx);
    err |= clSetKernelArg(context->kernels.outer_source, 1, sizeof(unsigned int), &rankinfo->ny);
    err |= clSetKernelArg(context->kernels.outer_source, 2, sizeof(unsigned int), &rankinfo->nz);
    err |= clSetKernelArg(context->kernels.outer_source, 3, sizeof(unsigned int), &problem->ng);
    err |= clSetKernelArg(context->kernels.outer_source, 4, sizeof(unsigned int), &problem->cmom);
    err |= clSetKernelArg(context->kernels.outer_source, 5, sizeof(unsigned int), &problem->nmom);
    err |= clSetKernelArg(context->kernels.outer_source, 6, sizeof(cl_mem), &buffers->fixed_source);
    err |= clSetKernelArg(context->kernels.outer_source, 7, sizeof(cl_mem), &buffers->scattering_matrix);
    err |= clSetKernelArg(context->kernels.outer_source, 8, sizeof(cl_mem), &buffers->scalar_flux);
    err |= clSetKernelArg(context->kernels.outer_source, 9, sizeof(cl_mem), &buffers->scalar_flux_moments);
    err |= clSetKernelArg(context->kernels.outer_source, 10, sizeof(cl_mem), &buffers->outer_source);
    check_ocl(err, "Setting outer source kernel arguments");

    size_t global[] = {rankinfo->nx, rankinfo->ny, rankinfo->nz};
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.outer_source,
        3, 0, global, NULL,
        0, NULL, NULL);
    check_ocl(err, "Enqueue outer source kernel");
}


void compute_inner_source(
    const struct problem * global,
    const struct rankinfo * rankinfo,
    const double * restrict outer_source,
    const double * restrict scattering_matrix,
    const double * restrict scalar_flux,
    const double * restrict scalar_flux_moments,
    double * restrict inner_source
    )
{
    for (unsigned int k = 0; k < rankinfo->nz; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
                for (unsigned int g = 0; g < global->ng; g++)
                {
                    // Set first moment to outer source plus scattering contribution of scalar flux
                    inner_source[SOURCE_INDEX(0,g,i,j,k,global->cmom,global->ng,rankinfo->nx,rankinfo->ny)]
                        = outer_source[SOURCE_INDEX(0,g,i,j,k,global->cmom,global->ng,rankinfo->nx,rankinfo->ny)]
                        + (scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g,global->nmom,global->ng)]
                        * scalar_flux[SCALAR_FLUX_INDEX(g,i,j,k,global->ng,rankinfo->nx,rankinfo->ny)]);

                    // Set other moments similarly based on scalar flux moments
                    unsigned int mom = 1;
                    for (unsigned int l = 0; l < global->nmom; l++)
                    {
                        for (unsigned int m = 0; m < 2*l+1; m++)
                        {
                            inner_source[SOURCE_INDEX(mom,g,i,j,k,global->cmom,global->ng,rankinfo->nx,rankinfo->ny)]
                                = outer_source[SOURCE_INDEX(mom,g,i,j,k,global->cmom,global->ng,rankinfo->nx,rankinfo->ny)] +
                                + (scattering_matrix[SCATTERING_MATRIX_INDEX(l,g,g,global->nmom,global->ng)]
                                * scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX(mom-1,g,i,j,k,global->cmom,global->ng,rankinfo->nx,rankinfo->ny)]);
                            mom += 1;
                        }
                    }
                }
}
