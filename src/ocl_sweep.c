
#include "ocl_sweep.h"

void sweep_plane(
    const int octant,
    const int istep,
    const int jstep,
    const int kstep,
    const unsigned int plane,
    const struct plane * planes,
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct context * context,
    struct buffers * buffers
    )
{
    cl_int err;

    // 2 dimensional kernel
    // First dimension: number of angles * number of groups
    // Second dimension: number of cells in plane
    size_t global[] = {problem->nang*problem->ng, planes[plane].num_cells};

    // Set the (many) kernel arguments
    err = clSetKernelArg(context->kernels.sweep_plane, 0, sizeof(unsigned int), &rankinfo->nx);
    err |= clSetKernelArg(context->kernels.sweep_plane, 1, sizeof(unsigned int), &rankinfo->ny);
    err |= clSetKernelArg(context->kernels.sweep_plane, 2, sizeof(unsigned int), &rankinfo->nz);
    err |= clSetKernelArg(context->kernels.sweep_plane, 3, sizeof(unsigned int), &problem->nang);
    err |= clSetKernelArg(context->kernels.sweep_plane, 4, sizeof(unsigned int), &problem->ng);
    err |= clSetKernelArg(context->kernels.sweep_plane, 5, sizeof(unsigned int), &problem->cmom);
    err |= clSetKernelArg(context->kernels.sweep_plane, 6, sizeof(int), &istep);
    err |= clSetKernelArg(context->kernels.sweep_plane, 7, sizeof(int), &jstep);
    err |= clSetKernelArg(context->kernels.sweep_plane, 8, sizeof(int), &kstep);
    err |= clSetKernelArg(context->kernels.sweep_plane, 9, sizeof(int), &octant);
    err |= clSetKernelArg(context->kernels.sweep_plane, 10, sizeof(cl_mem), &buffers->planes[plane]);
    err |= clSetKernelArg(context->kernels.sweep_plane, 11, sizeof(cl_mem), &buffers->inner_source);
    err |= clSetKernelArg(context->kernels.sweep_plane, 12, sizeof(cl_mem), &buffers->scat_coeff);
    err |= clSetKernelArg(context->kernels.sweep_plane, 13, sizeof(cl_mem), &buffers->dd_i);
    err |= clSetKernelArg(context->kernels.sweep_plane, 14, sizeof(cl_mem), &buffers->dd_j);
    err |= clSetKernelArg(context->kernels.sweep_plane, 15, sizeof(cl_mem), &buffers->dd_k);
    err |= clSetKernelArg(context->kernels.sweep_plane, 16, sizeof(cl_mem), &buffers->mu);
    err |= clSetKernelArg(context->kernels.sweep_plane, 17, sizeof(cl_mem), &buffers->velocity_delta);
    err |= clSetKernelArg(context->kernels.sweep_plane, 18, sizeof(cl_mem), &buffers->denominator);
    err |= clSetKernelArg(context->kernels.sweep_plane, 19, sizeof(cl_mem), &buffers->angular_flux_in[octant]);
    err |= clSetKernelArg(context->kernels.sweep_plane, 20, sizeof(cl_mem), &buffers->flux_i);
    err |= clSetKernelArg(context->kernels.sweep_plane, 21, sizeof(cl_mem), &buffers->flux_j);
    err |= clSetKernelArg(context->kernels.sweep_plane, 22, sizeof(cl_mem), &buffers->flux_k);
    err |= clSetKernelArg(context->kernels.sweep_plane, 23, sizeof(cl_mem), &buffers->angular_flux_out[octant]);

    check_ocl(err, "Setting plane sweep kernel arguments");

    // Actually enqueue
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.sweep_plane,
        2, 0, global, NULL,
        0, NULL, NULL);
    check_ocl(err, "Enqueue plane sweep kernel");
}
