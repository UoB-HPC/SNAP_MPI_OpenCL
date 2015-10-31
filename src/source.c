
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
        0, NULL, &outer_source_event);
    check_ocl(err, "Enqueue outer source kernel");
}


void compute_inner_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    struct context * context,
    struct buffers * buffers
    )
{
    cl_int err;
    err = clSetKernelArg(context->kernels.inner_source, 0, sizeof(unsigned int), &rankinfo->nx);
    err |= clSetKernelArg(context->kernels.inner_source, 1, sizeof(unsigned int), &rankinfo->ny);
    err |= clSetKernelArg(context->kernels.inner_source, 2, sizeof(unsigned int), &rankinfo->nz);
    err |= clSetKernelArg(context->kernels.inner_source, 3, sizeof(unsigned int), &problem->ng);
    err |= clSetKernelArg(context->kernels.inner_source, 4, sizeof(unsigned int), &problem->cmom);
    err |= clSetKernelArg(context->kernels.inner_source, 5, sizeof(unsigned int), &problem->nmom);
    err |= clSetKernelArg(context->kernels.inner_source, 6, sizeof(cl_mem), &buffers->outer_source);
    err |= clSetKernelArg(context->kernels.inner_source, 7, sizeof(cl_mem), &buffers->scattering_matrix);
    err |= clSetKernelArg(context->kernels.inner_source, 8, sizeof(cl_mem), &buffers->scalar_flux);
    err |= clSetKernelArg(context->kernels.inner_source, 9, sizeof(cl_mem), &buffers->scalar_flux_moments);
    err |= clSetKernelArg(context->kernels.inner_source, 10, sizeof(cl_mem), &buffers->inner_source);
    check_ocl(err, "Setting inner source kernel arguments");

    size_t global[] = {rankinfo->nx, rankinfo->ny, rankinfo->nz};
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.inner_source,
        3, 0, global, NULL,
        0, NULL, &inner_source_event);
    check_ocl(err, "Enqueue inner source kernel");
}
