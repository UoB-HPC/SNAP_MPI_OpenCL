
#include "scalar_flux.h"

void compute_scalar_flux(
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct context * context,
    struct buffers * buffers
    )
{

    // get smallest power of 2 greater than nang
    size_t power = 1 << (unsigned int)ceil(log2((double)problem->nang));

    const size_t global[] = {power * problem->ng, rankinfo->nx*rankinfo->ny*rankinfo->nz};
    const size_t local[] = {power, 1};

    cl_int err;
    err  = clSetKernelArg(context->kernels.reduce_flux,  0, sizeof(unsigned int), &rankinfo->nx);
    err |= clSetKernelArg(context->kernels.reduce_flux,  1, sizeof(unsigned int), &rankinfo->ny);
    err |= clSetKernelArg(context->kernels.reduce_flux,  2, sizeof(unsigned int), &rankinfo->nz);
    err |= clSetKernelArg(context->kernels.reduce_flux,  3, sizeof(unsigned int), &problem->nang);
    err |= clSetKernelArg(context->kernels.reduce_flux,  4, sizeof(unsigned int), &problem->ng);
    err |= clSetKernelArg(context->kernels.reduce_flux,  5, sizeof(cl_mem), &buffers->angular_flux_in[0]);
    err |= clSetKernelArg(context->kernels.reduce_flux,  6, sizeof(cl_mem), &buffers->angular_flux_in[1]);
    err |= clSetKernelArg(context->kernels.reduce_flux,  7, sizeof(cl_mem), &buffers->angular_flux_in[2]);
    err |= clSetKernelArg(context->kernels.reduce_flux,  8, sizeof(cl_mem), &buffers->angular_flux_in[3]);
    err |= clSetKernelArg(context->kernels.reduce_flux,  9, sizeof(cl_mem), &buffers->angular_flux_in[4]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 10, sizeof(cl_mem), &buffers->angular_flux_in[5]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 11, sizeof(cl_mem), &buffers->angular_flux_in[6]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 12, sizeof(cl_mem), &buffers->angular_flux_in[7]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 13, sizeof(cl_mem), &buffers->angular_flux_out[0]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 14, sizeof(cl_mem), &buffers->angular_flux_out[1]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 15, sizeof(cl_mem), &buffers->angular_flux_out[2]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 16, sizeof(cl_mem), &buffers->angular_flux_out[3]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 17, sizeof(cl_mem), &buffers->angular_flux_out[4]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 18, sizeof(cl_mem), &buffers->angular_flux_out[5]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 19, sizeof(cl_mem), &buffers->angular_flux_out[6]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 20, sizeof(cl_mem), &buffers->angular_flux_out[7]);
    err |= clSetKernelArg(context->kernels.reduce_flux, 21, sizeof(cl_mem), &buffers->velocity_delta);
    err |= clSetKernelArg(context->kernels.reduce_flux, 22, sizeof(cl_mem), &buffers->quad_weights);
    err |= clSetKernelArg(context->kernels.reduce_flux, 23, sizeof(cl_mem), &buffers->scalar_flux);
    err |= clSetKernelArg(context->kernels.reduce_flux, 24, sizeof(double)*local[0], NULL);
    check_ocl(err, "Setting scalar flux reduction kernel arguments");

    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.reduce_flux,
        2, 0, global, local,
        0, NULL, NULL);
    check_ocl(err, "Enqueueing scalar flux reduction kernel");

}

void copy_back_scalar_flux(
    struct problem *problem,
    struct rankinfo * rankinfo,
    struct context * context,
    struct buffers * buffers,
    double * scalar_flux,
    cl_bool blocking
    )
{
    cl_int err;
    err = clEnqueueReadBuffer(context->queue, buffers->scalar_flux, blocking,
        0, sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz, scalar_flux,
        0, NULL, NULL);
    check_ocl(err, "Copying back scalar flux");
}
